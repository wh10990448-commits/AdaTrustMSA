"""Build a complete, auditable CMU-MOSEI dataset from computational sequences.

The extractor enumerates the label CSD first, so a sample is never silently
discarded merely because words, audio, or vision are absent. Missing modality
observations are represented by zeros plus explicit observation masks. Official
train/valid/test folds are read directly from CMU-MultimodalSDK.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import csv
import hashlib
import json
from pathlib import Path
import runpy
from typing import Iterable

import h5py
import numpy as np
from transformers import AutoTokenizer

from prepare_data import fit_statistics, save_json


EXPECTED_DIMENSIONS = {"audio": 74, "vision": 35, "labels": 7}
EXPECTED_FILES = {
    "labels": "CMU_MOSEI_Labels.csd",
    "words": "CMU_MOSEI_TimestampedWords.csd",
    "audio": "CMU_MOSEI_COVAREP.csd",
    "vision": "CMU_MOSEI_VisualFacet42.csd",
}
LABEL_NAMES = ("sentiment", "happy", "sad", "anger", "surprise", "disgust", "fear")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def open_data_group(stack: ExitStack, path: Path):
    handle = stack.enter_context(h5py.File(path, "r"))
    roots = [key for key in handle if isinstance(handle[key], h5py.Group) and "data" in handle[key]]
    if len(roots) != 1:
        raise ValueError(f"{path}: expected exactly one computational-sequence root, found {roots}")
    return handle[roots[0]]["data"], roots[0]


def locate_fold_file(sdk_root: Path) -> Path:
    matches = list(sdk_root.rglob("cmu_mosei_std_folds.py"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one cmu_mosei_std_folds.py below {sdk_root}, found {len(matches)}"
        )
    return matches[0]


def load_official_folds(sdk_root: Path) -> tuple[dict[str, set[str]], Path]:
    fold_file = locate_fold_file(sdk_root)
    namespace = runpy.run_path(str(fold_file))
    folds = {
        "train": set(namespace["standard_train_fold"]),
        "valid": set(namespace["standard_valid_fold"]),
        "test": set(namespace["standard_test_fold"]),
    }
    if any(folds[left] & folds[right] for left, right in (("train", "valid"), ("train", "test"), ("valid", "test"))):
        raise ValueError("official fold file contains overlapping video IDs")
    return folds, fold_file


def decode_word(value) -> str:
    value = np.asarray(value).reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def validate_sequence(video_id: str, group, expected_dim: int | None, name: str):
    if video_id not in group:
        return None, None
    node = group[video_id]
    features = np.asarray(node["features"])
    intervals = np.asarray(node["intervals"], dtype=np.float64)
    if features.ndim != 2 or intervals.shape != (len(features), 2):
        raise ValueError(
            f"{name}/{video_id}: incompatible shapes features={features.shape}, intervals={intervals.shape}"
        )
    if expected_dim is not None and features.shape[1] != expected_dim:
        raise ValueError(
            f"{name}/{video_id}: feature dimension {features.shape[1]}, expected {expected_dim}"
        )
    if len(intervals) and (not np.isfinite(intervals).all() or np.any(intervals[:, 1] < intervals[:, 0])):
        raise ValueError(f"{name}/{video_id}: invalid time intervals")
    return features, intervals


def overlapping_rows(intervals: np.ndarray, start: float, end: float) -> tuple[int, int]:
    """Return a half-open slice of sorted rows whose intervals overlap [start, end]."""
    if not len(intervals):
        return 0, 0
    left = int(np.searchsorted(intervals[:, 1], start, side="right"))
    right = int(np.searchsorted(intervals[:, 0], end, side="left"))
    return left, max(left, right)


def finite_mean(features: np.ndarray, left: int, right: int, dimension: int):
    if right <= left:
        return np.zeros(dimension, dtype=np.float32), False, 0
    block = np.asarray(features[left:right], dtype=np.float64)
    finite = np.isfinite(block)
    counts = finite.sum(axis=0)
    sums = np.where(finite, block, 0.0).sum(axis=0)
    value = np.divide(sums, counts, out=np.zeros(dimension), where=counts > 0)
    return value.astype(np.float32), bool(np.any(counts > 0)), int((~finite).sum())


def aligned_modality(
    features: np.ndarray | None,
    intervals: np.ndarray | None,
    word_interval: np.ndarray,
    dimension: int,
):
    if features is None or intervals is None:
        return np.zeros(dimension, dtype=np.float32), False, 0
    left, right = overlapping_rows(intervals, float(word_interval[0]), float(word_interval[1]))
    return finite_mean(features, left, right, dimension)


def empty_bucket() -> dict[str, list]:
    return defaultdict(list)


def append_sample(bucket: dict[str, list], sample: dict) -> None:
    for key, value in sample.items():
        bucket[key].append(value)


def stack_bucket(bucket: dict[str, list]) -> dict[str, np.ndarray]:
    if not bucket:
        return {}
    text_fields = {"sample_id", "video_id"}
    output = {}
    for key, values in bucket.items():
        if key in text_fields:
            output[key] = np.asarray(values, dtype=str)
        else:
            output[key] = np.stack(values)
    return output


def write_manifest(path: Path, rows: list[dict]) -> None:
    fields = [
        "sample_id",
        "video_id",
        "segment_index",
        "split",
        "official_split",
        "start",
        "end",
        "word_count",
        "subword_count",
        "text_observed",
        "audio_observed",
        "vision_observed",
        "truncated",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_sample(
    *,
    video_id: str,
    segment_index: int,
    label_vector: np.ndarray,
    label_interval: np.ndarray,
    word_features: np.ndarray | None,
    word_intervals: np.ndarray | None,
    audio_features: np.ndarray | None,
    audio_intervals: np.ndarray | None,
    vision_features: np.ndarray | None,
    vision_intervals: np.ndarray | None,
    tokenizer,
    sequence_length: int,
):
    max_content = sequence_length - 2
    ids: list[int] = []
    audio_rows: list[np.ndarray] = []
    vision_rows: list[np.ndarray] = []
    audio_observed: list[int] = []
    vision_observed: list[int] = []
    nonfinite = Counter()
    words_used = 0
    truncated = False

    if word_features is not None and word_intervals is not None:
        left, right = overlapping_rows(
            word_intervals, float(label_interval[0]), float(label_interval[1])
        )
        for word_index in range(left, right):
            word = decode_word(word_features[word_index])
            if not word:
                continue
            pieces = tokenizer.tokenize(word)
            if not pieces:
                pieces = [tokenizer.unk_token]
            piece_ids = tokenizer.convert_tokens_to_ids(pieces)
            if len(ids) + len(piece_ids) > max_content:
                piece_ids = piece_ids[: max_content - len(ids)]
                truncated = True
            if not piece_ids:
                break
            a_value, a_seen, a_bad = aligned_modality(
                audio_features,
                audio_intervals,
                word_intervals[word_index],
                EXPECTED_DIMENSIONS["audio"],
            )
            v_value, v_seen, v_bad = aligned_modality(
                vision_features,
                vision_intervals,
                word_intervals[word_index],
                EXPECTED_DIMENSIONS["vision"],
            )
            nonfinite["audio"] += a_bad
            nonfinite["vision"] += v_bad
            ids.extend(piece_ids)
            audio_rows.extend([a_value] * len(piece_ids))
            vision_rows.extend([v_value] * len(piece_ids))
            audio_observed.extend([int(a_seen)] * len(piece_ids))
            vision_observed.extend([int(v_seen)] * len(piece_ids))
            words_used += 1
            if len(ids) == max_content:
                if word_index + 1 < right:
                    truncated = True
                break

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id
    if cls_id is None or sep_id is None or pad_id is None:
        raise ValueError("tokenizer must define CLS, SEP, and PAD token IDs")
    input_ids = np.full(sequence_length, pad_id, dtype=np.int64)
    attention = np.zeros(sequence_length, dtype=np.int64)
    token_types = np.zeros(sequence_length, dtype=np.int64)
    audio = np.zeros((sequence_length, EXPECTED_DIMENSIONS["audio"]), dtype=np.float32)
    vision = np.zeros((sequence_length, EXPECTED_DIMENSIONS["vision"]), dtype=np.float32)
    audio_mask = np.zeros(sequence_length, dtype=np.uint8)
    vision_mask = np.zeros(sequence_length, dtype=np.uint8)

    full_ids = [cls_id, *ids, sep_id]
    used = len(full_ids)
    input_ids[:used] = full_ids
    attention[:used] = 1
    if ids:
        audio[1 : 1 + len(ids)] = np.stack(audio_rows)
        vision[1 : 1 + len(ids)] = np.stack(vision_rows)
        audio_mask[1 : 1 + len(ids)] = audio_observed
        vision_mask[1 : 1 + len(ids)] = vision_observed

    sentiment = float(label_vector[0])
    sample = {
        "input_ids": input_ids,
        "attention_mask": attention,
        "token_type_ids": token_types,
        "audio": audio,
        "vision": vision,
        "audio_observed": audio_mask,
        "vision_observed": vision_mask,
        "classification_labels": np.asarray(int(np.sign(sentiment)) + 1, dtype=np.int64),
        "regression_labels": np.asarray(sentiment, dtype=np.float32),
        "emotion_labels": np.asarray(label_vector[1:7], dtype=np.float32),
        "label_vector": np.asarray(label_vector, dtype=np.float32),
        "sample_id": f"{video_id}[{segment_index}]",
        "video_id": video_id,
        "segment_index": np.asarray(segment_index, dtype=np.int32),
        "segment_interval": np.asarray(label_interval, dtype=np.float32),
    }
    facts = {
        "word_count": words_used,
        "subword_count": len(ids),
        "text_observed": bool(ids),
        "audio_observed": bool(np.any(audio_mask)),
        "vision_observed": bool(np.any(vision_mask)),
        "truncated": truncated,
        "nonfinite_audio_values": nonfinite["audio"],
        "nonfinite_vision_values": nonfinite["vision"],
    }
    return sample, facts


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--csd-dir", type=Path, required=True)
    result.add_argument("--sdk-root", type=Path, required=True)
    result.add_argument("--tokenizer", required=True, help="Local path or Hugging Face tokenizer ID")
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--sequence-length", type=int, default=50)
    result.add_argument(
        "--unassigned-policy",
        choices=("separate", "train", "drop"),
        default="separate",
        help="How to handle labeled videos absent from the official folds",
    )
    result.add_argument(
        "--strict-trimodal",
        action="store_true",
        help="Drop samples lacking observed text, audio, or vision (not recommended for full extraction)",
    )
    result.add_argument(
        "--expected-label-rows",
        type=int,
        default=0,
        help="Fail unless the label CSD has this exact row count; 0 disables the assertion",
    )
    result.add_argument("--local-files-only", action="store_true")
    result.add_argument("--skip-source-hash", action="store_true")
    return result


def main(argv: Iterable[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.sequence_length < 4:
        raise ValueError("sequence length must be at least 4")
    paths = {name: args.csd_dir / filename for name, filename in EXPECTED_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required CSD files: {missing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    folds, fold_file = load_official_folds(args.sdk_root)
    video_to_split = {}
    for split, video_ids in folds.items():
        video_to_split.update({video_id: split for video_id in video_ids})
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        use_fast=True,
        local_files_only=args.local_files_only,
    )

    buckets = {"train": empty_bucket(), "valid": empty_bucket(), "test": empty_bucket()}
    if args.unassigned_policy == "separate":
        buckets["unassigned"] = empty_bucket()
    audit = Counter()
    manifest_rows: list[dict] = []
    label_videos = set()
    roots = {}

    with ExitStack() as stack:
        labels, roots["labels"] = open_data_group(stack, paths["labels"])
        words, roots["words"] = open_data_group(stack, paths["words"])
        audio, roots["audio"] = open_data_group(stack, paths["audio"])
        vision, roots["vision"] = open_data_group(stack, paths["vision"])
        source_rows = sum(labels[video_id]["features"].shape[0] for video_id in labels)
        official_fold_source_rows = {
            split: int(
                sum(
                    labels[video_id]["features"].shape[0]
                    for video_id in video_ids
                    if video_id in labels
                )
            )
            for split, video_ids in folds.items()
        }
        if args.expected_label_rows and source_rows != args.expected_label_rows:
            raise ValueError(
                f"label CSD contains {source_rows} rows, expected {args.expected_label_rows}; "
                "do not continue until the source version is understood"
            )

        for video_number, video_id in enumerate(sorted(labels), start=1):
            label_videos.add(video_id)
            label_features, label_intervals = validate_sequence(
                video_id, labels, EXPECTED_DIMENSIONS["labels"], "labels"
            )
            word_features, word_intervals = validate_sequence(video_id, words, 1, "words")
            audio_features, audio_intervals = validate_sequence(
                video_id, audio, EXPECTED_DIMENSIONS["audio"], "audio"
            )
            vision_features, vision_intervals = validate_sequence(
                video_id, vision, EXPECTED_DIMENSIONS["vision"], "vision"
            )
            original_split = video_to_split.get(video_id, "unassigned")
            output_split = original_split
            if original_split == "unassigned":
                audit["unassigned_source_rows"] += len(label_features)
                if args.unassigned_policy == "train":
                    output_split = "train"
                elif args.unassigned_policy == "drop":
                    audit["dropped_unassigned"] += len(label_features)
                    continue

            for segment_index, (label_vector, label_interval) in enumerate(
                zip(label_features, label_intervals)
            ):
                audit["label_rows_visited"] += 1
                if not np.isfinite(label_vector).all():
                    raise ValueError(f"labels/{video_id}[{segment_index}]: nonfinite label")
                if not -3.0 <= float(label_vector[0]) <= 3.0:
                    raise ValueError(f"labels/{video_id}[{segment_index}]: sentiment outside [-3,3]")
                sample, facts = build_sample(
                    video_id=video_id,
                    segment_index=segment_index,
                    label_vector=label_vector,
                    label_interval=label_interval,
                    word_features=word_features,
                    word_intervals=word_intervals,
                    audio_features=audio_features,
                    audio_intervals=audio_intervals,
                    vision_features=vision_features,
                    vision_intervals=vision_intervals,
                    tokenizer=tokenizer,
                    sequence_length=args.sequence_length,
                )
                for modality in ("text", "audio", "vision"):
                    if facts[f"{modality}_observed"]:
                        audit[f"observed_{modality}"] += 1
                    else:
                        audit[f"missing_{modality}"] += 1
                audit["truncated"] += int(facts["truncated"])
                audit["word_count"] += int(facts["word_count"])
                audit["subword_count"] += int(facts["subword_count"])
                audit["nonfinite_audio_values"] += int(facts["nonfinite_audio_values"])
                audit["nonfinite_vision_values"] += int(facts["nonfinite_vision_values"])
                complete = facts["text_observed"] and facts["audio_observed"] and facts["vision_observed"]
                if args.strict_trimodal and not complete:
                    audit["dropped_incomplete"] += 1
                    continue
                append_sample(buckets[output_split], sample)
                audit["emitted_samples"] += 1
                audit[
                    "emitted_unassigned" if original_split == "unassigned" else "emitted_official_fold"
                ] += 1
                manifest_rows.append(
                    {
                        "sample_id": sample["sample_id"],
                        "video_id": video_id,
                        "segment_index": segment_index,
                        "split": output_split,
                        "official_split": original_split,
                        "start": float(label_interval[0]),
                        "end": float(label_interval[1]),
                        **{key: facts[key] for key in (
                            "word_count",
                            "subword_count",
                            "text_observed",
                            "audio_observed",
                            "vision_observed",
                            "truncated",
                        )},
                    }
                )
            if video_number % 100 == 0 or video_number == len(labels):
                print(
                    f"processed videos {video_number}/{len(labels)}, emitted {audit['emitted_samples']}",
                    flush=True,
                )

    accounted_rows = (
        audit["emitted_samples"]
        + audit["dropped_unassigned"]
        + audit["dropped_incomplete"]
    )
    if accounted_rows != source_rows:
        raise RuntimeError(
            f"coverage invariant failed: accounted for {accounted_rows} of {source_rows} label rows"
        )

    arrays = {}
    for split, bucket in buckets.items():
        data = stack_bucket(bucket)
        if not data:
            continue
        arrays[split] = data
        np.savez_compressed(args.output_dir / f"{split}.npz", **data)
    if "train" not in arrays or "valid" not in arrays or "test" not in arrays:
        raise ValueError("the official train/valid/test splits must all contain samples")

    statistics, normalization_audit = fit_statistics(arrays["train"])
    save_json(args.output_dir / "preprocessing.json", statistics)
    write_manifest(args.output_dir / "sample_manifest.csv", manifest_rows)

    source_hashes = {}
    if not args.skip_source_hash:
        for name, path in paths.items():
            print(f"hashing {path.name}", flush=True)
            source_hashes[name] = sha256(path)
    emitted_splits = {name: int(len(data["input_ids"])) for name, data in arrays.items()}
    coverage = {
        "dataset": "CMU-MOSEI",
        "format_version": "adatrustmsa-csd-v1",
        "source_label_rows": int(source_rows),
        "source_label_videos": len(label_videos),
        "official_fold_video_counts": {name: len(ids) for name, ids in folds.items()},
        "official_fold_source_rows": official_fold_source_rows,
        "official_fold_source_total": int(sum(official_fold_source_rows.values())),
        "emitted_splits": emitted_splits,
        "emitted_samples": int(audit["emitted_samples"]),
        "accounted_source_rows": int(accounted_rows),
        "all_source_rows_accounted_for": bool(accounted_rows == source_rows),
        "emitted_official_fold_rows": int(audit["emitted_official_fold"]),
        "emitted_unassigned_rows": int(audit["emitted_unassigned"]),
        "unassigned_policy": args.unassigned_policy,
        "strict_trimodal": args.strict_trimodal,
        "missing_segments": {
            "text": int(audit["missing_text"]),
            "audio": int(audit["missing_audio"]),
            "vision": int(audit["missing_vision"]),
        },
        "dropped": {
            "unassigned": int(audit["dropped_unassigned"]),
            "incomplete": int(audit["dropped_incomplete"]),
        },
        "truncated_segments": int(audit["truncated"]),
        "nonfinite_source_values_replaced": {
            "audio": int(audit["nonfinite_audio_values"]),
            "vision": int(audit["nonfinite_vision_values"]),
        },
        "sequence_length": args.sequence_length,
        "feature_dimensions": {"audio": 74, "vision": 35},
        "label_names": list(LABEL_NAMES),
        "csd_roots": roots,
        "sources": {name: str(path.resolve()) for name, path in paths.items()},
        "source_sha256": source_hashes,
        "official_fold_file": str(fold_file.resolve()),
        "official_fold_file_sha256": sha256(fold_file),
        "tokenizer": args.tokenizer,
        "normalization": normalization_audit,
        "integrity_statement": (
            "Every row in the supplied label CSD was enumerated. Missing observations were "
            "not synthesized; they remain zero with explicit observation masks. Counts describe "
            "this exact CSD version and must not be replaced by a paper-level corpus claim."
        ),
    }
    contract = {
        "dataset": "CMU-MOSEI",
        "dataset_key": "mosei",
        "feature_version": "official-computational-sequences-word-aligned-v1",
        "feature_contract": {
            "text": f"{args.tokenizer} token IDs; word timestamps from CMU_MOSEI_TimestampedWords",
            "audio": "CMU_MOSEI_COVAREP, 74 dimensions, mean-pooled per word",
            "vision": "CMU_MOSEI_VisualFacet42, 35 dimensions, mean-pooled per word",
            "alignment": "label interval -> word interval -> overlapping modality frames",
            "sequence_length": args.sequence_length,
            "missing_value_policy": "zero plus audio_observed/vision_observed mask; never imputed",
        },
        "label_mapping": {"0": "Negative", "1": "Neutral", "2": "Positive"},
        "splits": {name: {"n": count, "labels_cached": True} for name, count in emitted_splits.items()},
        "total_samples": int(audit["emitted_samples"]),
        "protocol": (
            "extended-all-label" if args.unassigned_policy == "train" else "official-folds-plus-audited-unassigned"
        ),
    }
    save_json(args.output_dir / "coverage_audit.json", coverage)
    save_json(args.output_dir / "data_contract.json", contract)
    print(json.dumps(coverage, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
