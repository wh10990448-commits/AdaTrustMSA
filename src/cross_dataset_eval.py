"""Zero-shot cross-dataset evaluation, e.g. CMU-MOSEI to CMU-MOSI.

The checkpoint, normalization statistics and adaptive policy are all selected
on the source dataset. Target labels are used only once for final reporting.
No domain adaptation is performed in this baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from adatrustmsa_corruption import INPUT_KEYS
from adatrustmsa_metrics import metrics
from adatrustmsa_policy import anytime_policy, predict_numpy
from evaluate import load_model, policy_metrics, write_csv
from train import load_npz, save_json


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def validate_target_contract(
    data: dict,
    contract: dict,
    split: str,
    expected_audio_dim: int,
    expected_vision_dim: int,
) -> None:
    ids = data["input_ids"]
    found = (data["audio"].shape[-1], data["vision"].shape[-1])
    expected = (expected_audio_dim, expected_vision_dim)
    if found != expected:
        raise ValueError(
            "source and target feature spaces are incompatible: "
            f"target A/V dimensions are {found}, checkpoint expects {expected}. "
            "Re-extract both datasets with the same audio/vision feature "
            "configuration, or implement an explicit domain adapter; do not "
            "pad unrelated features."
        )
    contract_count = contract.get("splits", {}).get(split, {}).get("n")
    if contract_count is not None and int(contract_count) != len(ids):
        raise ValueError("target NPZ count disagrees with data_contract.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-data-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--selected-policy",
        required=True,
        type=Path,
        help="Policy JSON selected on the source validation split",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", default="test", choices=("valid", "test"))
    parser.add_argument("--source-dataset", default="CMU-MOSEI")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()

    policy = read_json(args.selected_policy)
    if policy.get("selected_on") != "valid":
        raise ValueError(
            "selected-policy must come from a source validation run "
            "(expected selected_on='valid')"
        )
    contract = read_json(args.target_data_root / "data_contract.json")
    target_data = load_npz(args.target_data_root, args.split)
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, payload = load_model(args.checkpoint, device)
    validate_target_contract(
        target_data,
        contract,
        args.split,
        expected_audio_dim=int(model.audio_mean.numel()),
        expected_vision_dim=int(model.vision_mean.numel()),
    )
    config = payload["config"]
    inputs = {key: target_data[key] for key in INPUT_KEYS}
    labels = target_data["classification_labels"]
    regression_labels = target_data["regression_labels"]

    full_output = predict_numpy(
        model, inputs, device, batch_size=int(config["eval_batch_size"])
    )
    full_metrics = metrics(
        labels,
        regression_labels,
        full_output["logits"],
        full_output["regression"],
    )

    policy_config = config["policy"]
    rows, resource_summary = anytime_policy(
        model,
        inputs,
        device,
        prefixes=policy_config["prefixes"],
        modality_order=policy_config["modality_order"],
        modality_threshold=float(
            policy.get("modality_threshold", policy_config["modality_threshold"])
        ),
        exit_threshold=float(policy["exit_threshold"]),
        minimum_exit_prefix=float(policy_config["minimum_exit_prefix"]),
        require_temporal_stability=bool(policy_config["require_temporal_stability"]),
        batch_size=int(config["eval_batch_size"]),
    )
    adaptive_metrics = policy_metrics(labels, regression_labels, rows)

    sample_ids = target_data.get("sample_id", np.arange(len(rows)))
    for sample_id, true_class, true_intensity, row in zip(
        sample_ids, labels, regression_labels, rows
    ):
        row["sample_id"] = str(sample_id)
        row["true_class"] = int(true_class)
        row["true_intensity"] = float(true_intensity)
        row["correct"] = int(row["predicted_class"] == int(true_class))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "predictions.csv", rows)
    result = {
        "protocol": "zero-shot cross-dataset evaluation",
        "source_dataset": args.source_dataset,
        "target_dataset": contract.get("dataset", "unknown"),
        "target_split": args.split,
        "target_samples": int(len(labels)),
        "checkpoint": str(args.checkpoint),
        "selected_policy": str(args.selected_policy),
        "normalization": "source checkpoint statistics",
        "domain_adaptation": "none",
        "target_labels_used_for_training_or_selection": False,
        "full_TAV_metrics": full_metrics,
        "adaptive_metrics": adaptive_metrics,
        "adaptive_resource_summary": resource_summary,
    }
    save_json(args.output_dir / "cross_dataset_evaluation.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
