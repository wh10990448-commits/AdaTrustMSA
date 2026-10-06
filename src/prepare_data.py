"""Convert standard MMSA CMU-MOSEI or CMU-MOSI pickles into NPZ files.

All splits retain labels. Normalization statistics are fitted on the selected
dataset's training observations only. The resulting data contract records the
dataset identity and feature dimensions for reproducible in-domain and
cross-dataset evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np


DATASETS = {
    "mosei": {
        "name": "CMU-MOSEI",
        "splits": {"train": 16326, "valid": 1871, "test": 4659},
        "standard_total": 22856,
        "reported_segments": 23453,
        "audio_dimension": 74,
        "vision_dimension": 35,
    },
    "mosi": {
        "name": "CMU-MOSI",
        "splits": {"train": 1284, "valid": 229, "test": 686},
        "standard_total": 2199,
        "reported_segments": 2199,
        "audio_dimension": 5,
        "vision_dimension": 20,
    },
}
SPECIAL_IDS = (101, 102)


def save_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unpack_text_bert(values, split: str):
    array = np.asarray(values)
    if array.ndim != 3:
        raise ValueError(f"{split}: text_bert must be 3-dimensional")
    if array.shape[1] == 3:
        channels = array
    elif array.shape[2] == 3:
        channels = array.transpose(0, 2, 1)
    else:
        raise ValueError(f"{split}: text_bert must have layout (N,3,T) or (N,T,3)")
    if not np.isfinite(channels).all():
        raise ValueError(f"{split}: text_bert contains nonfinite values")
    integer = channels.astype(np.int64)
    if not np.array_equal(channels, integer):
        raise ValueError(f"{split}: text_bert contains fractional IDs")
    ids, attention, token_types = (integer[:, index] for index in range(3))
    if not np.isin(attention, (0, 1)).all():
        raise ValueError(f"{split}: attention_mask must be binary")
    if ids.min() < 0 or ids.max() >= 30522:
        raise ValueError(f"{split}: BERT IDs outside bert-base-uncased vocabulary")
    if token_types.min() < 0 or token_types.max() >= 2:
        raise ValueError(f"{split}: token type IDs outside expected range")
    if np.any(attention.sum(axis=1) < 2):
        raise ValueError(f"{split}: found an empty text sequence")
    return ids, attention, token_types


def convert_split(
    block: dict,
    split: str,
    expected_audio_dim: int = 74,
    expected_vision_dim: int = 35,
) -> dict:
    required = ("text_bert", "audio", "vision", "regression_labels")
    missing = [key for key in required if key not in block]
    if missing:
        raise KeyError(f"{split}: missing fields {missing}")
    ids, attention, token_types = unpack_text_bert(block["text_bert"], split)
    count, sequence_length = ids.shape
    audio = np.asarray(block["audio"], dtype=np.float32)
    vision = np.asarray(block["vision"], dtype=np.float32)
    if audio.shape != (count, sequence_length, expected_audio_dim):
        raise ValueError(
            f"{split}: audio shape {audio.shape}, expected "
            f"{(count, sequence_length, expected_audio_dim)}"
        )
    if vision.shape != (count, sequence_length, expected_vision_dim):
        raise ValueError(
            f"{split}: vision shape {vision.shape}, expected "
            f"{(count, sequence_length, expected_vision_dim)}"
        )
    if not np.isfinite(audio).all() or not np.isfinite(vision).all():
        raise ValueError(f"{split}: audio or vision contains nonfinite values")

    regression = np.asarray(block["regression_labels"], dtype=np.float32).reshape(-1)
    if len(regression) != count or not np.isfinite(regression).all():
        raise ValueError(f"{split}: invalid regression labels")
    if regression.min() < -3 or regression.max() > 3:
        raise ValueError(f"{split}: regression labels must be in [-3,3]")
    derived_class = (np.sign(regression).astype(np.int64) + 1).reshape(-1)
    if "classification_labels" in block:
        supplied = np.asarray(block["classification_labels"], dtype=np.int64).reshape(-1)
        if len(supplied) != count:
            raise ValueError(f"{split}: classification label length mismatch")
        if not np.array_equal(supplied, derived_class):
            raise ValueError(f"{split}: class labels disagree with regression-label sign")
        classification = supplied
    else:
        classification = derived_class

    if "id" in block:
        sample_id = np.asarray([str(value) for value in np.asarray(block["id"]).reshape(-1)])
    else:
        sample_id = np.asarray([f"{split}_{index:06d}" for index in range(count)])
    if len(sample_id) != count or len(set(sample_id.tolist())) != count:
        raise ValueError(f"{split}: sample IDs are missing or duplicated")
    return {
        "input_ids": ids.astype(np.int64),
        "attention_mask": attention.astype(np.int64),
        "token_type_ids": token_types.astype(np.int64),
        "audio": audio,
        "vision": vision,
        "classification_labels": classification,
        "regression_labels": regression,
        "sample_id": sample_id,
    }


def fit_statistics(train: dict):
    content = (
        train["attention_mask"].astype(bool)
        & (train["input_ids"] != SPECIAL_IDS[0])
        & (train["input_ids"] != SPECIAL_IDS[1])
    )
    statistics = {}
    audit = {}
    for key in ("audio", "vision"):
        observation_key = f"{key}_observed"
        if observation_key in train:
            observed = np.asarray(train[observation_key], dtype=bool)
            if observed.shape != content.shape:
                raise ValueError(
                    f"{observation_key} shape {observed.shape}, expected {content.shape}"
                )
        else:
            observed = np.any(train[key] != 0, axis=-1)
        available = content & observed
        values = train[key][available].astype(np.float64)
        if not len(values):
            raise ValueError(f"training split has no observed {key} rows")
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        constant = std < 1e-6
        std[constant] = 1.0
        statistics[f"{key}_mean"] = mean.astype(np.float32).tolist()
        statistics[f"{key}_std"] = std.astype(np.float32).tolist()
        audit[f"{key}_normalization"] = {
            "fit_split": "train",
            "observed_rows": int(len(values)),
            "constant_dimensions": np.flatnonzero(constant).tolist(),
            "post_standardization_clip": [-10, 10],
        }
    return statistics, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=tuple(DATASETS))
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="MMSA aligned_50.pkl for the selected dataset",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--allow-nonstandard-size",
        action="store_true",
        help="Permit split counts that differ from the selected standard dataset",
    )
    args = parser.parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.input.open("rb") as handle:
        raw = pickle.load(handle)
    if not isinstance(raw, dict):
        raise TypeError("dataset pickle must contain a split dictionary")

    specification = DATASETS[args.dataset]
    expected_splits = specification["splits"]
    converted = {}
    all_ids = set()
    summary = {
        "dataset": specification["name"],
        "dataset_key": args.dataset,
        "source": str(args.input.resolve()),
        "source_sha256": sha256(args.input),
        "feature_version": "aligned_50",
        "expected_standard_splits": expected_splits,
        "feature_contract": {
            "text": "bert-base-uncased IDs, mask and token types",
            "audio_dimension": specification["audio_dimension"],
            "vision_dimension": specification["vision_dimension"],
            "alignment": "word-aligned",
        },
        "label_mapping": {"0": "Negative", "1": "Neutral", "2": "Positive"},
        "splits": {},
    }
    for split in ("train", "valid", "test"):
        if split not in raw:
            raise KeyError(f"dataset has no {split!r} split")
        data = convert_split(
            raw[split],
            split,
            expected_audio_dim=specification["audio_dimension"],
            expected_vision_dim=specification["vision_dimension"],
        )
        count = len(data["input_ids"])
        if not args.allow_nonstandard_size and count != expected_splits[split]:
            raise ValueError(
                f"{split}: found {count} samples, expected {expected_splits[split]}; "
                "use --allow-nonstandard-size only when this is intentional"
            )
        ids = set(data["sample_id"].tolist())
        if ids & all_ids:
            raise ValueError(f"{split}: sample IDs overlap an earlier split")
        all_ids |= ids
        np.savez_compressed(args.output_dir / f"{split}.npz", **data)
        summary["splits"][split] = {
            "n": count,
            "class_counts": np.bincount(data["classification_labels"], minlength=3).tolist(),
            "sequence_length": int(data["input_ids"].shape[1]),
            "labels_cached": True,
        }
        converted[split] = data

    statistics, normalization_audit = fit_statistics(converted["train"])
    summary.update(normalization_audit)
    summary["total_samples"] = int(sum(item["n"] for item in summary["splits"].values()))
    summary["standard_usable_total"] = specification["standard_total"]
    summary["original_corpus_reported_segments"] = specification["reported_segments"]
    summary["note"] = (
        "AdaTrustMSA uses standard MMSA benchmark splits for comparable experiments. "
        "Cross-dataset evaluation keeps source-checkpoint normalization to avoid target leakage."
    )
    save_json(args.output_dir / "preprocessing.json", statistics)
    save_json(args.output_dir / "data_contract.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
