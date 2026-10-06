"""Evaluate AdaTrustMSA robustness, calibration and resource-aware policies."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

from train import load_npz, save_json
from adatrustmsa_corruption import INPUT_KEYS, corrupt_batch, disable_modalities
from adatrustmsa_metrics import consistent_predictions, metrics, selective_curve
from adatrustmsa_model import AdaTrustMSAModel
from adatrustmsa_policy import anytime_policy, predict_numpy


def load_model(path: Path, device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model_kwargs = dict(payload["model_kwargs"])
    legacy_v1 = "fusion_layers" not in model_kwargs
    if legacy_v1:
        model_kwargs["fusion_layers"] = 0
    model = AdaTrustMSAModel(**model_kwargs)
    model.load_state_dict(payload["state_dict"], strict=not legacy_v1)
    return model.to(device).eval(), payload


def write_csv(path: Path, rows: list) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def policy_metrics(labels, regression_labels, rows):
    predicted = np.array([row["predicted_class"] for row in rows], np.int64)
    intensity = np.array([row["predicted_intensity"] for row in rows], np.float32)
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, labels=[0, 1, 2], average="macro", zero_division=0)),
        "mae": float(mean_absolute_error(regression_labels, intensity)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", default="valid", choices=("valid", "test"))
    parser.add_argument(
        "--selected-policy",
        type=Path,
        help="Required for test: policy JSON selected on the validation split",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    data = load_npz(args.data_root, args.split)
    model, payload = load_model(args.checkpoint, device)
    config = payload["config"]
    inputs = {key: data[key] for key in INPUT_KEYS}
    labels = data["classification_labels"]
    regression_labels = data["regression_labels"]

    fixed_policy = None
    if args.selected_policy is not None:
        fixed_policy = json.loads(args.selected_policy.read_text(encoding="utf-8"))
        if fixed_policy.get("selected_on") != "valid":
            raise ValueError("selected policy must have selected_on='valid'")
    if args.split == "test" and fixed_policy is None:
        raise ValueError("--selected-policy is required when --split test")

    clean_output = predict_numpy(model, inputs, device, int(config["eval_batch_size"]))
    clean_metrics = metrics(labels, regression_labels, clean_output["logits"], clean_output["regression"])
    predicted, _ = consistent_predictions(clean_output["logits"], clean_output["regression"])
    selective = selective_curve(labels, predicted, clean_output["confidence"])

    robustness_rows = []
    for enabled in ("T", "A", "V", "TA", "TV", "AV", "TAV"):
        changed = disable_modalities(inputs, tuple(enabled))
        output = predict_numpy(model, changed, device, int(config["eval_batch_size"]))
        result = metrics(labels, regression_labels, output["logits"], output["regression"])
        robustness_rows.append({"scenario": f"enabled_{enabled}", **result})

    reliability_rows = []
    for kind_index, (kind, gaussian_probability) in enumerate(
        (("occlusion", 0.0), ("gaussian", 1.0))
    ):
        for ratio_index, ratio in enumerate((0.1, 0.25, 0.5, 0.75)):
            rng = np.random.default_rng(42000 + kind_index * 100 + ratio_index)
            changed, target, audit = corrupt_batch(
                inputs,
                rng,
                probability=1.0,
                ratio_range=(ratio, ratio),
                gaussian_probability=gaussian_probability,
                modality_subsets=("A", "V", "AV"),
                audio_std=model.audio_std.detach().cpu().numpy(),
                vision_std=model.vision_std.detach().cpu().numpy(),
            )
            output = predict_numpy(model, changed, device, int(config["eval_batch_size"]))
            result = metrics(labels, regression_labels, output["logits"], output["regression"])
            reliability_mae = np.abs(output["reliability"] - target).mean(axis=0)
            row = {
                "scenario": f"av_{kind}_{ratio:.2f}",
                "requested_ratio": ratio,
                "corruption_kind": kind,
                "missing_samples": int(sum(item.get("kind") == "missing" for item in audit)),
                "gaussian_samples": int(sum(item.get("kind") == "gaussian" for item in audit)),
                "reliability_mae_T": float(reliability_mae[0]),
                "reliability_mae_A": float(reliability_mae[1]),
                "reliability_mae_V": float(reliability_mae[2]),
                **result,
            }
            robustness_rows.append(row)
            reliability_rows.append(row)

    policy_cfg = config["policy"]
    modality_threshold = float(
        fixed_policy.get("modality_threshold", policy_cfg["modality_threshold"])
        if fixed_policy is not None
        else policy_cfg["modality_threshold"]
    )
    grid = []
    policy_outputs = {}
    thresholds = (
        [float(fixed_policy["exit_threshold"])]
        if fixed_policy is not None
        else policy_cfg["exit_threshold_candidates"]
    )
    for threshold in thresholds:
        rows, summary = anytime_policy(
            model,
            inputs,
            device,
            prefixes=policy_cfg["prefixes"],
            modality_order=policy_cfg["modality_order"],
            modality_threshold=modality_threshold,
            exit_threshold=float(threshold),
            minimum_exit_prefix=float(policy_cfg["minimum_exit_prefix"]),
            require_temporal_stability=bool(policy_cfg["require_temporal_stability"]),
            batch_size=int(config["eval_batch_size"]),
        )
        result = policy_metrics(labels, regression_labels, rows)
        entry = {"exit_threshold": float(threshold), **summary, **result}
        grid.append(entry)
        policy_outputs[float(threshold)] = rows

    if fixed_policy is not None:
        selected = grid[0]
        reason = "fixed policy selected on the source validation split"
    else:
        allowable = [row for row in grid if row["macro_f1"] >= clean_metrics["macro_f1"] - 0.02]
        if allowable:
            selected = min(
                allowable,
                key=lambda row: (row["mean_exit_prefix"] + 0.2 * row["mean_modalities_at_exit"] / 3, -row["macro_f1"]),
            )
            reason = "minimum resource score subject to validation Macro-F1 drop <= 0.02"
        else:
            selected = max(grid, key=lambda row: row["macro_f1"])
            reason = "no threshold met the F1 constraint; selected highest validation Macro-F1"
    selected_rows = policy_outputs[selected["exit_threshold"]]
    sample_ids = data.get("sample_id", np.arange(len(selected_rows)))
    for sample_id, true_class, true_intensity, row in zip(
        sample_ids, labels, regression_labels, selected_rows
    ):
        row["sample_id"] = str(sample_id)
        row["true_class"] = int(true_class)
        row["true_intensity"] = float(true_intensity)
        row["correct"] = int(row["predicted_class"] == int(true_class))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "robustness_grid.csv", robustness_rows)
    write_csv(args.output_dir / "policy_grid.csv", grid)
    write_csv(args.output_dir / "selected_policy_predictions.csv", selected_rows)
    save_json(
        args.output_dir / "evaluation.json",
        {
            "checkpoint": str(args.checkpoint),
            "split": args.split,
            "clean": clean_metrics,
            "selective_prediction": selective,
            "selected_policy": {**selected, "selection_reason": reason},
            "policy_grid": grid,
            "robustness_scenarios": len(robustness_rows),
            "labels_used_for_policy_selection": "valid",
        },
    )
    policy_name = "selected_policy.json" if fixed_policy is None else "applied_policy.json"
    save_json(
        args.output_dir / policy_name,
        {
            **selected,
            "selection_reason": reason,
            "selected_on": "valid",
            "evaluated_on": args.split,
        },
    )
    print(json.dumps({"clean": clean_metrics, "selected_policy": selected}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
