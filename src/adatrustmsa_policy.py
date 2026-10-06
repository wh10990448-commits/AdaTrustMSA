"""Dynamic modality calling and temporal early-exit policy for AdaTrustMSA."""

from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
import time
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from adatrustmsa_corruption import INPUT_KEYS, apply_prefix
from adatrustmsa_metrics import consistent_predictions


RETURN_KEYS = (
    "logits",
    "regression",
    "confidence",
    "variance",
    "reliability",
    "modality_weights",
    "modality_conflict",
)


def _subset(data: Mapping[str, np.ndarray], indices: np.ndarray) -> dict:
    return {key: np.asarray(value)[indices] for key, value in data.items()}


@torch.inference_mode()
def predict_numpy(model, data, device, batch_size=128) -> dict:
    model.eval()
    collected = {key: [] for key in RETURN_KEYS}
    count = len(data["input_ids"])
    for start in range(0, count, batch_size):
        stop = min(count, start + batch_size)
        batch = {
            key: torch.as_tensor(data[key][start:stop], device=device)
            for key in INPUT_KEYS
        }
        context = (
            torch.autocast("cuda", dtype=torch.float16)
            if device.type == "cuda"
            else nullcontext()
        )
        with context:
            output = model(**batch)
        for key in RETURN_KEYS:
            collected[key].append(output[key].float().cpu().numpy())
    return {key: np.concatenate(values, axis=0) for key, values in collected.items()}


def _allocate_current(size: int) -> dict:
    return {
        "logits": np.zeros((size, 3), np.float32),
        "regression": np.zeros(size, np.float32),
        "confidence": np.zeros(size, np.float32),
        "variance": np.zeros(size, np.float32),
        "reliability": np.zeros((size, 3), np.float32),
        "modality_weights": np.zeros((size, 3), np.float32),
        "modality_conflict": np.zeros((size, 3), np.float32),
    }


@torch.inference_mode()
def progressive_predict_numpy(
    model,
    data: Mapping[str, np.ndarray],
    device,
    modality_order: Sequence[str],
    modality_threshold: float,
    batch_size: int = 128,
) -> tuple[dict, np.ndarray, float]:
    """Acquire new modality branches without recomputing earlier branches."""
    model.eval()
    count = len(data["input_ids"])
    combined = _allocate_current(count)
    stages_used = np.zeros(count, np.int64)
    modality_index = {"T": 0, "A": 1, "V": 2}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()

    for start in range(0, count, batch_size):
        stop = min(count, start + batch_size)
        batch = {
            key: torch.as_tensor(data[key][start:stop], device=device)
            for key in INPUT_KEYS
        }
        size = stop - start
        pooled = torch.zeros(
            size, 3, model.output_dim, device=device, dtype=torch.float32
        )
        fractions = torch.zeros(size, 3, device=device, dtype=torch.float32)
        present = torch.zeros(size, 3, device=device, dtype=torch.bool)
        need_more = torch.ones(size, device=device, dtype=torch.bool)

        for stage, modality in enumerate(modality_order, start=1):
            indices = torch.nonzero(need_more, as_tuple=False).flatten()
            if indices.numel() == 0:
                break
            selected = {key: value.index_select(0, indices) for key, value in batch.items()}
            context = (
                torch.autocast("cuda", dtype=torch.float16)
                if device.type == "cuda"
                else nullcontext()
            )
            with context:
                state, fraction, is_present = model.encode_modality(
                    modality, **selected
                )
                slot = modality_index[modality]
                pooled[indices, slot] = state.float()
                fractions[indices, slot] = fraction.float()
                present[indices, slot] = is_present
                output = model.predict_from_pooled(
                    pooled.index_select(0, indices),
                    fractions.index_select(0, indices),
                    present.index_select(0, indices),
                )
            local_indices = indices.cpu().numpy()
            global_indices = start + local_indices
            for key in combined:
                combined[key][global_indices] = output[key].float().cpu().numpy()
            stages_used[global_indices] = stage
            need_more[indices] = output["confidence"] < modality_threshold

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return combined, stages_used, elapsed


def anytime_policy(
    model,
    data: Mapping[str, np.ndarray],
    device,
    prefixes: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
    modality_order: Iterable[str] = ("T", "A", "V"),
    modality_threshold: float = 0.60,
    exit_threshold: float = 0.68,
    minimum_exit_prefix: float = 0.5,
    require_temporal_stability: bool = True,
    batch_size: int = 128,
) -> tuple[list, dict]:
    """Run text-first dynamic acquisition followed by temporal early exit.

    Each prefix first uses the cheapest modality, and only adds the next modality
    when confidence is below ``modality_threshold``.  Temporal exit additionally
    requires stable class decisions across two consecutive prefixes by default.
    """
    prefixes = tuple(float(value) for value in prefixes)
    if not prefixes or any(value <= 0 or value > 1 for value in prefixes):
        raise ValueError("prefixes must be nonempty values in (0,1]")
    if tuple(sorted(prefixes)) != prefixes or prefixes[-1] != 1.0:
        raise ValueError("prefixes must be sorted and end in 1.0")
    order = tuple(item.upper() for item in modality_order)
    if not order or len(set(order)) != len(order) or not set(order).issubset({"T", "A", "V"}):
        raise ValueError("modality_order must contain unique T/A/V symbols")
    if not 0 <= modality_threshold <= 1 or not 0 <= exit_threshold <= 1:
        raise ValueError("thresholds must be in [0,1]")

    sample_count = len(data["input_ids"])
    unresolved = np.arange(sample_count, dtype=np.int64)
    previous_class = np.full(sample_count, -1, np.int64)
    cumulative_calls = np.zeros(sample_count, np.int64)
    cumulative_encoded_branches = np.zeros(sample_count, np.int64)
    total_inference_seconds = 0.0
    records = [None] * sample_count

    for prefix in prefixes:
        if not len(unresolved):
            break
        prefixed = apply_prefix(data, prefix)
        active_data = _subset(prefixed, unresolved)
        local_size = len(unresolved)
        current, stages_used, elapsed = progressive_predict_numpy(
            model,
            active_data,
            device,
            modality_order=order,
            modality_threshold=modality_threshold,
            batch_size=batch_size,
        )
        total_inference_seconds += elapsed
        cumulative_calls[unresolved] += stages_used
        cumulative_encoded_branches[unresolved] += stages_used

        predicted_class, intensity = consistent_predictions(
            current["logits"], current["regression"]
        )
        stable = previous_class[unresolved] == predicted_class
        eligible = current["confidence"] >= exit_threshold
        eligible &= prefix >= minimum_exit_prefix
        if require_temporal_stability and prefix < 1.0:
            eligible &= stable
        if prefix == 1.0:
            eligible[:] = True

        exiting_local = np.flatnonzero(eligible)
        for local in exiting_local:
            global_index = int(unresolved[local])
            used = order[: int(stages_used[local])]
            records[global_index] = {
                "sample_index": global_index,
                "predicted_class": int(predicted_class[local]),
                "predicted_intensity": float(intensity[local]),
                "confidence": float(current["confidence"][local]),
                "variance": float(current["variance"][local]),
                "exit_prefix": float(prefix),
                "modalities_used": "".join(used),
                "modality_count": int(stages_used[local]),
                "cumulative_model_calls": int(cumulative_calls[global_index]),
                "cumulative_encoded_branches": int(
                    cumulative_encoded_branches[global_index]
                ),
                "reliability_T": float(current["reliability"][local, 0]),
                "reliability_A": float(current["reliability"][local, 1]),
                "reliability_V": float(current["reliability"][local, 2]),
                "weight_T": float(current["modality_weights"][local, 0]),
                "weight_A": float(current["modality_weights"][local, 1]),
                "weight_V": float(current["modality_weights"][local, 2]),
                "conflict_T": float(current["modality_conflict"][local, 0]),
                "conflict_A": float(current["modality_conflict"][local, 1]),
                "conflict_V": float(current["modality_conflict"][local, 2]),
            }

        previous_class[unresolved] = predicted_class
        unresolved = unresolved[~eligible]

    if any(record is None for record in records):
        raise RuntimeError("policy failed to produce a prediction for every sample")
    prefix_distribution = Counter(str(row["exit_prefix"]) for row in records)
    route_distribution = Counter(row["modalities_used"] for row in records)
    summary = {
        "samples": sample_count,
        "mean_exit_prefix": float(np.mean([row["exit_prefix"] for row in records])),
        "early_exit_rate": float(np.mean([row["exit_prefix"] < 1.0 for row in records])),
        "mean_modalities_at_exit": float(np.mean([row["modality_count"] for row in records])),
        "mean_cumulative_model_calls": float(
            np.mean([row["cumulative_model_calls"] for row in records])
        ),
        "mean_encoded_modality_branches": float(
            np.mean([row["cumulative_encoded_branches"] for row in records])
        ),
        "cache_reuse_within_prefix": True,
        "measured_inference_seconds": float(total_inference_seconds),
        "measured_milliseconds_per_sample": float(
            1000.0 * total_inference_seconds / max(sample_count, 1)
        ),
        "exit_prefix_distribution": dict(prefix_distribution),
        "modality_route_distribution": dict(route_distribution),
        "modality_order": list(order),
        "modality_threshold": float(modality_threshold),
        "exit_threshold": float(exit_threshold),
        "minimum_exit_prefix": float(minimum_exit_prefix),
        "require_temporal_stability": bool(require_temporal_stability),
    }
    return records, summary
