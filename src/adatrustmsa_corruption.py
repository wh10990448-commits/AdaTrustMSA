"""Deterministic complex-scene simulation for aligned T/A/V sequences.

The functions never mutate their inputs.  They use the project convention:
text is removed with BERT UNK, audio/vision with zeros, and padding stays fixed.
Known corruption strength is returned as a reliability target in [0, 1].
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence, Tuple

import numpy as np


PAD_ID, UNK_ID, CLS_ID, SEP_ID = 0, 100, 101, 102
CHANNELS = ("T", "A", "V")
INPUT_KEYS = ("input_ids", "attention_mask", "token_type_ids", "audio", "vision")


def content_mask(input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    ids = np.asarray(input_ids)
    attention = np.asarray(attention_mask)
    if ids.ndim != 2 or attention.shape != ids.shape:
        raise ValueError("input_ids and attention_mask must have shape (N,L)")
    return (attention != 0) & (ids != CLS_ID) & (ids != SEP_ID)


def observed_fractions(data: Mapping[str, np.ndarray]) -> np.ndarray:
    content = content_mask(data["input_ids"], data["attention_mask"])
    text = content & (data["input_ids"] != PAD_ID) & (data["input_ids"] != UNK_ID)
    audio = content & np.any(data["audio"] != 0, axis=-1)
    vision = content & np.any(data["vision"] != 0, axis=-1)
    denominator = np.maximum(content.sum(axis=1, keepdims=True), 1)
    return np.stack((text, audio, vision), axis=-1).sum(axis=1) / denominator


def _copy_inputs(data: Mapping[str, np.ndarray]) -> dict:
    missing = [key for key in INPUT_KEYS if key not in data]
    if missing:
        raise KeyError(f"missing input keys: {missing}")
    copied = {key: np.array(data[key], copy=True) for key in data}
    ids = copied["input_ids"]
    if copied["attention_mask"].shape != ids.shape or copied["token_type_ids"].shape != ids.shape:
        raise ValueError("BERT arrays must have matching (N,L) shapes")
    if copied["audio"].shape != (*ids.shape, 74):
        raise ValueError("audio must have shape (N,L,74)")
    if copied["vision"].shape != (*ids.shape, 35):
        raise ValueError("vision must have shape (N,L,35)")
    return copied


def disable_modalities(
    data: Mapping[str, np.ndarray], enabled: Iterable[str]
) -> dict:
    """Return inputs containing only the requested modalities."""
    enabled = {item.upper() for item in enabled}
    if not enabled.issubset(CHANNELS):
        raise ValueError(f"enabled modalities must be a subset of {CHANNELS}")
    result = _copy_inputs(data)
    content = content_mask(result["input_ids"], result["attention_mask"])
    if "T" not in enabled:
        result["input_ids"][content] = UNK_ID
    if "A" not in enabled:
        result["audio"][content] = 0
    if "V" not in enabled:
        result["vision"][content] = 0
    return result


def apply_prefix(data: Mapping[str, np.ndarray], fraction: float) -> dict:
    """Keep the first fraction of content and right-pad the remainder."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0,1]")
    result = _copy_inputs(data)
    if fraction == 1:
        return result
    content = content_mask(result["input_ids"], result["attention_mask"])
    for row in range(len(content)):
        positions = np.flatnonzero(content[row])
        if not len(positions):
            continue
        keep = max(1, int(np.ceil(len(positions) * fraction)))
        if keep >= len(positions):
            continue
        cutoff = int(positions[keep - 1]) + 1
        result["input_ids"][row, cutoff:] = PAD_ID
        result["attention_mask"][row, cutoff:] = 0
        result["token_type_ids"][row, cutoff:] = 0
        result["audio"][row, cutoff:] = 0
        result["vision"][row, cutoff:] = 0
    return result


def corrupt_batch(
    data: Mapping[str, np.ndarray],
    rng: np.random.Generator,
    probability: float = 0.7,
    ratio_range: Tuple[float, float] = (0.1, 0.6),
    gaussian_probability: float = 0.35,
    cascade_dropout_probability: float = 0.0,
    modality_subsets: Sequence[str] = ("T", "A", "V", "TA", "TV", "AV", "TAV"),
    audio_std: Sequence[float] | None = None,
    vision_std: Sequence[float] | None = None,
) -> Tuple[dict, np.ndarray, list]:
    """Apply per-sample local corruption and return reliability targets.

    A sample selects one configured nonempty modality subset. Text uses a local
    UNK span. Audio/vision use either a zero span or feature-scaled Gaussian
    noise. At least one content position is retained per modality.
    """
    if not 0 <= probability <= 1:
        raise ValueError("probability must be in [0,1]")
    low, high = map(float, ratio_range)
    if not 0 <= low <= high <= 1:
        raise ValueError("ratio_range must satisfy 0 <= low <= high <= 1")
    if not 0 <= gaussian_probability <= 1:
        raise ValueError("gaussian_probability must be in [0,1]")
    if not 0 <= cascade_dropout_probability <= 1:
        raise ValueError("cascade_dropout_probability must be in [0,1]")
    subsets = tuple(item.upper() for item in modality_subsets)
    if not subsets or any(not item or not set(item).issubset(CHANNELS) for item in subsets):
        raise ValueError("modality_subsets must contain nonempty T/A/V combinations")
    result = _copy_inputs(data)
    content = content_mask(result["input_ids"], result["attention_mask"])
    original_fraction = observed_fractions(result).astype(np.float32)
    reliability = original_fraction.copy()
    audit = []
    audio_scale = np.ones(74, np.float32) if audio_std is None else np.asarray(audio_std, np.float32)
    vision_scale = np.ones(35, np.float32) if vision_std is None else np.asarray(vision_std, np.float32)
    if audio_scale.shape != (74,) or vision_scale.shape != (35,):
        raise ValueError("audio_std/vision_std have incorrect shapes")

    for row in range(len(content)):
        positions = np.flatnonzero(content[row])
        if len(positions) < 2 or rng.random() >= probability:
            audit.append({"applied": False, "channels": "", "ratio": 0.0, "kind": "clean"})
            continue
        channels = subsets[int(rng.integers(len(subsets)))]
        ratio = float(rng.uniform(low, high))
        length = min(len(positions) - 1, max(1, int(np.ceil(ratio * len(positions)))))
        start = int(rng.integers(0, len(positions) - length + 1))
        span = positions[start : start + length]
        noise = rng.random() < gaussian_probability and any(c in channels for c in "AV")

        if "T" in channels:
            result["input_ids"][row, span] = UNK_ID
        if "A" in channels:
            if noise:
                observed_span = span[np.any(result["audio"][row, span] != 0, axis=-1)]
                if len(observed_span):
                    perturb = rng.normal(size=(len(observed_span), 74)).astype(np.float32)
                    severity = max(ratio, 0.05)
                    result["audio"][row, observed_span] += perturb * audio_scale * severity
                    reliability[row, 1] *= (
                        1.0 - 0.75 * severity * len(observed_span) / len(positions)
                    )
            else:
                result["audio"][row, span] = 0
        if "V" in channels:
            if noise:
                observed_span = span[np.any(result["vision"][row, span] != 0, axis=-1)]
                if len(observed_span):
                    perturb = rng.normal(size=(len(observed_span), 35)).astype(np.float32)
                    severity = max(ratio, 0.05)
                    result["vision"][row, observed_span] += perturb * vision_scale * severity
                    reliability[row, 2] *= (
                        1.0 - 0.75 * severity * len(observed_span) / len(positions)
                    )
            else:
                result["vision"][row, span] = 0
        audit.append(
            {
                "applied": True,
                "channels": channels,
                "ratio": ratio,
                "span_length": int(length),
                "kind": "gaussian" if noise else "missing",
            }
        )

    # Text corruption is always UNK-based, even when A/V in the same sample use
    # Gaussian noise. Missing-span A/V targets use actual remaining support.
    post_fraction = observed_fractions(result).astype(np.float32)
    reliability[:, 0] = post_fraction[:, 0]
    for index, channel in enumerate(CHANNELS[1:], start=1):
        missing_rows = np.array(
            [entry.get("kind") == "missing" and channel in entry.get("channels", "") for entry in audit]
        )
        reliability[missing_rows, index] = post_fraction[missing_rows, index]

    # Train the same T -> TA -> TAV states used by adaptive inference. This is
    # intentionally separate from local noise/occlusion simulation.
    for row in range(len(content)):
        if rng.random() >= cascade_dropout_probability:
            audit[row]["cascade_route"] = "TAV"
            continue
        route = "T" if rng.random() < 0.5 else "TA"
        positions = content[row]
        if route == "T":
            result["audio"][row, positions] = 0
            reliability[row, 1] = 0.0
        result["vision"][row, positions] = 0
        reliability[row, 2] = 0.0
        audit[row]["cascade_route"] = route
    reliability = np.clip(reliability, 0.0, 1.0).astype(np.float32)
    return result, reliability, audit


def self_test() -> dict:
    rng = np.random.default_rng(7)
    ids = np.array(
        [[101, 110, 111, 112, 113, 102, 0], [101, 120, 121, 102, 0, 0, 0]],
        dtype=np.int64,
    )
    attention = (ids != 0).astype(np.int64)
    data = {
        "input_ids": ids,
        "attention_mask": attention,
        "token_type_ids": np.zeros_like(ids),
        "audio": np.ones((*ids.shape, 74), np.float32),
        "vision": np.ones((*ids.shape, 35), np.float32),
    }
    data["audio"][~content_mask(ids, attention)] = 0
    data["vision"][~content_mask(ids, attention)] = 0
    changed, target, audit = corrupt_batch(
        data, rng, probability=1.0, ratio_range=(0.5, 0.5), gaussian_probability=0.0
    )
    assert target.shape == (2, 3) and np.all((0 <= target) & (target <= 1))
    assert not np.shares_memory(changed["audio"], data["audio"])
    assert all(entry["applied"] for entry in audit)
    prefixed = apply_prefix(data, 0.5)
    assert np.all(prefixed["attention_mask"].sum(axis=1) <= attention.sum(axis=1))
    text_only = disable_modalities(data, ("T",))
    assert not text_only["audio"].any() and not text_only["vision"].any()
    np.testing.assert_array_equal(data["input_ids"], ids)
    return {"status": "PASS", "samples": 2, "targets": target.tolist()}


if __name__ == "__main__":
    print(self_test())
