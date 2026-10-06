"""Standalone AdaTrustMSA inference with uncertainty and early exit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import pickle
import re

import numpy as np
import torch

from adatrustmsa_corruption import INPUT_KEYS
from adatrustmsa_model import AdaTrustMSAModel
from adatrustmsa_policy import anytime_policy


CLASS_NAMES = ("Negative", "Neutral", "Positive")


def load_model(path: Path, device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model_kwargs = dict(payload["model_kwargs"])
    legacy_v1 = "fusion_layers" not in model_kwargs
    if legacy_v1:
        model_kwargs["fusion_layers"] = 0
    model = AdaTrustMSAModel(**model_kwargs)
    model.load_state_dict(payload["state_dict"], strict=not legacy_v1)
    return model.to(device).eval(), payload


def file_key(path: Path):
    match = re.search(r"(\d+)$", path.stem)
    return (0, int(match.group(1))) if match else (1, path.name)


def load_samples(directory: Path, vocab_size: int, type_vocab_size: int):
    files = sorted(directory.glob("*.pkl"), key=file_key)
    if not files:
        raise FileNotFoundError(f"No .pkl inputs found in {directory}")
    samples = []
    for path in files:
        with path.open("rb") as handle:
            obj = pickle.load(handle)
        if "test" not in obj:
            raise KeyError(f"{path.name}: expected top-level 'test' block")
        block = obj["test"]
        text_bert = np.asarray(block["text_bert"])
        if text_bert.shape != (1, 3, 50) or not np.isfinite(text_bert).all():
            raise ValueError(f"{path.name}: text_bert must be finite with shape (1,3,50)")
        integer = text_bert.astype(np.int64)
        if not np.array_equal(integer, text_bert):
            raise ValueError(f"{path.name}: text_bert contains fractional IDs")
        if not ((integer[:, 0] >= 0) & (integer[:, 0] < vocab_size)).all():
            raise ValueError(f"{path.name}: token ID outside model vocabulary")
        if not np.isin(integer[:, 1], (0, 1)).all():
            raise ValueError(f"{path.name}: attention mask must be binary")
        if not ((integer[:, 2] >= 0) & (integer[:, 2] < type_vocab_size)).all():
            raise ValueError(f"{path.name}: token type ID outside model vocabulary")
        audio = np.asarray(block["audio"], dtype=np.float32)
        vision = np.asarray(block["vision"], dtype=np.float32)
        if audio.shape != (1, 50, 74) or vision.shape != (1, 50, 35):
            raise ValueError(f"{path.name}: expected audio (1,50,74), vision (1,50,35)")
        if not np.isfinite(audio).all() or not np.isfinite(vision).all():
            raise ValueError(f"{path.name}: nonfinite audio or vision features")
        samples.append(
            {
                "input_ids": integer[:, 0],
                "attention_mask": integer[:, 1],
                "token_type_ids": integer[:, 2],
                "audio": audio,
                "vision": vision,
            }
        )
    data = {key: np.concatenate([sample[key] for sample in samples]) for key in INPUT_KEYS}
    return files, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--selected-policy", type=Path, help="Optional JSON produced by evaluate.py")
    parser.add_argument("--modality-threshold", type=float)
    parser.add_argument("--exit-threshold", type=float)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, payload = load_model(args.checkpoint, device)
    config = payload["config"]
    policy = dict(config["policy"])
    if args.selected_policy:
        selected = json.loads(args.selected_policy.read_text(encoding="utf-8"))
        if selected.get("selected_on") != "valid":
            raise ValueError("selected policy must have selected_on='valid'")
        policy["exit_threshold"] = selected["exit_threshold"]
        if "modality_threshold" in selected:
            policy["modality_threshold"] = selected["modality_threshold"]
    if args.modality_threshold is not None:
        policy["modality_threshold"] = args.modality_threshold
    if args.exit_threshold is not None:
        policy["exit_threshold"] = args.exit_threshold
    policy.setdefault("exit_threshold", 0.68)

    files, data = load_samples(
        args.input_dir, model.bert.config.vocab_size, model.bert.config.type_vocab_size
    )
    rows, summary = anytime_policy(
        model,
        data,
        device,
        prefixes=policy["prefixes"],
        modality_order=policy["modality_order"],
        modality_threshold=float(policy["modality_threshold"]),
        exit_threshold=float(policy["exit_threshold"]),
        minimum_exit_prefix=float(policy["minimum_exit_prefix"]),
        require_temporal_stability=bool(policy["require_temporal_stability"]),
        batch_size=int(config["eval_batch_size"]),
    )
    for path, row in zip(files, rows):
        row["sample_id"] = path.stem
        row["source_file"] = path.name
        row["predicted_sentiment"] = CLASS_NAMES[row["predicted_class"]]
    preferred = [
        "sample_id", "source_file", "predicted_sentiment", "predicted_class",
        "predicted_intensity", "confidence", "variance", "exit_prefix",
        "modalities_used", "modality_count", "cumulative_model_calls",
        "cumulative_encoded_branches",
        "reliability_T", "reliability_A", "reliability_V",
        "weight_T", "weight_A", "weight_V", "conflict_T", "conflict_A", "conflict_V",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=preferred)
        writer.writeheader()
        writer.writerows({key: row[key] for key in preferred} for row in rows)
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    print(f"Wrote {len(rows)} predictions to {args.output} using {device.type}")


if __name__ == "__main__":
    main()
