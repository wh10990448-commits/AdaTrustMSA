"""Select a two-checkpoint ensemble on validation data, then evaluate it once.

The ensemble mixes the continuous sentiment predictions of a regression-oriented
checkpoint and a classification-oriented checkpoint.  Selection maximises the
worst normalised validation margin over a declared reference result, which
prevents one easy metric from hiding a regression in another metric.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from adatrustmsa_corruption import INPUT_KEYS
from adatrustmsa_metrics import metrics
from adatrustmsa_policy import predict_numpy
from evaluate import load_model
from evaluate_paper import PAPER_KEYS
from train import load_npz, save_json


def _predict(checkpoint: Path, data: dict, device) -> tuple[dict, dict]:
    model, payload = load_model(checkpoint, device)
    output = predict_numpy(
        model,
        {key: data[key] for key in INPUT_KEYS},
        device,
        int(payload["config"]["eval_batch_size"]),
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output, payload


def _blend(
    first: dict,
    second: dict,
    second_weight: float,
    scale: float = 1.0,
    bias: float = 0.0,
    classification_head_weight: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    weight = float(second_weight)
    logits = (1.0 - weight) * first["logits"] + weight * second["logits"]
    regression = scale * (
        (1.0 - weight) * first["regression"] + weight * second["regression"]
    ) + bias
    if classification_head_weight > 0.0:
        shifted = logits - logits.max(axis=1, keepdims=True)
        probability = np.exp(shifted)
        probability /= probability.sum(axis=1, keepdims=True)
        classification_score = 3.0 * (probability[:, 2] - probability[:, 0])
        regression = (
            (1.0 - classification_head_weight) * regression
            + classification_head_weight * classification_score
        )
    return logits, regression


def _binary_weighted_f1(truth: np.ndarray, prediction: np.ndarray) -> float:
    total = len(truth)
    score = 0.0
    for label in (False, True):
        support = int(np.sum(truth == label))
        true_positive = int(np.sum((truth == label) & (prediction == label)))
        false_positive = int(np.sum((truth != label) & (prediction == label)))
        false_negative = int(np.sum((truth == label) & (prediction != label)))
        denominator = 2 * true_positive + false_positive + false_negative
        class_f1 = 0.0 if denominator == 0 else 2.0 * true_positive / denominator
        score += support * class_f1
    return float(score / max(total, 1))


def _fast_selection_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    nonzero = truth != 0
    nonzero_truth = truth[nonzero] > 0
    nonzero_prediction = prediction[nonzero] > 0
    return {
        "mmsa_acc_7": float(
            np.mean(
                np.round(np.clip(prediction, -3.0, 3.0))
                == np.round(np.clip(truth, -3.0, 3.0))
            )
        ),
        "mmsa_non0_acc_2": float(np.mean(nonzero_truth == nonzero_prediction)),
        "mmsa_non0_f1": _binary_weighted_f1(nonzero_truth, nonzero_prediction),
        "mmsa_mae": float(np.mean(np.abs(truth - prediction))),
    }


def _reference_margins(result: dict, reference: dict) -> dict:
    # Each value is a fractional improvement on the metric's natural scale.
    # MAE is reversed because lower is better.
    return {
        "acc_7": result["mmsa_acc_7"] - reference["acc_7"],
        "acc_2": result["mmsa_non0_acc_2"] - reference["acc_2"],
        "f1": result["mmsa_non0_f1"] - reference["f1"],
        "mae": reference["mae"] - result["mmsa_mae"],
    }


def _paper_metrics(data: dict, logits: np.ndarray, regression: np.ndarray) -> dict:
    result = metrics(
        data["classification_labels"],
        data["regression_labels"],
        logits,
        regression,
    )
    return {key: result[key] for key in PAPER_KEYS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--regression-checkpoint", required=True, type=Path)
    parser.add_argument("--classification-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--step", default=0.025, type=float)
    # Defaults reproduce the reported final result: checkpoint prediction
    # blending only. Scale/bias/head grids remain opt-in research options.
    parser.add_argument("--scale-min", default=1.0, type=float)
    parser.add_argument("--scale-max", default=1.0, type=float)
    parser.add_argument("--scale-step", default=0.01, type=float)
    parser.add_argument("--bias-min", default=0.0, type=float)
    parser.add_argument("--bias-max", default=0.0, type=float)
    parser.add_argument("--bias-step", default=0.005, type=float)
    parser.add_argument("--head-weight-max", default=0.0, type=float)
    parser.add_argument("--head-weight-step", default=0.025, type=float)
    parser.add_argument("--reference-acc7", default=0.541, type=float)
    parser.add_argument("--reference-acc2", default=0.853, type=float)
    parser.add_argument("--reference-f1", default=0.853, type=float)
    parser.add_argument("--reference-mae", default=0.536, type=float)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    if (
        not 0.0 < args.step <= 1.0
        or args.scale_step <= 0
        or args.bias_step <= 0
        or args.head_weight_step <= 0
        or not 0.0 <= args.head_weight_max <= 1.0
    ):
        raise ValueError("ensemble, scale and bias steps must be positive")

    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    reference = {
        "acc_7": args.reference_acc7,
        "acc_2": args.reference_acc2,
        "f1": args.reference_f1,
        "mae": args.reference_mae,
    }
    valid = load_npz(args.data_root, "valid")
    valid_regression, regression_payload = _predict(args.regression_checkpoint, valid, device)
    valid_classification, classification_payload = _predict(
        args.classification_checkpoint, valid, device
    )

    rows = []
    weights = np.unique(np.append(np.arange(0.0, 1.0 + args.step / 2.0, args.step), 1.0))
    scales = np.arange(args.scale_min, args.scale_max + args.scale_step / 2.0, args.scale_step)
    biases = np.arange(args.bias_min, args.bias_max + args.bias_step / 2.0, args.bias_step)
    head_weights = np.arange(
        0.0, args.head_weight_max + args.head_weight_step / 2.0, args.head_weight_step
    )
    for weight in weights:
        blended_logits = (
            (1.0 - float(weight)) * valid_regression["logits"]
            + float(weight) * valid_classification["logits"]
        )
        shifted = blended_logits - blended_logits.max(axis=1, keepdims=True)
        probability = np.exp(shifted)
        probability /= probability.sum(axis=1, keepdims=True)
        classification_score = 3.0 * (probability[:, 2] - probability[:, 0])
        raw_prediction = (
            (1.0 - float(weight)) * valid_regression["regression"]
            + float(weight) * valid_classification["regression"]
        )
        for scale in scales:
            for bias in biases:
                calibrated = float(scale) * raw_prediction + float(bias)
                for head_weight in head_weights:
                    regression = (
                        (1.0 - float(head_weight)) * calibrated
                        + float(head_weight) * classification_score
                    )
                    result = _fast_selection_metrics(valid["regression_labels"], regression)
                    margins = _reference_margins(result, reference)
                    rows.append(
                        {
                            "classification_weight": float(weight),
                            "regression_scale": float(scale),
                            "regression_bias": float(bias),
                            "classification_head_weight": float(head_weight),
                            "minimum_reference_margin": float(min(margins.values())),
                            "mean_reference_margin": float(np.mean(list(margins.values()))),
                            "reference_margins": margins,
                            "metrics": result,
                        }
                    )

    # Max-min selection is deliberately validation-only.  Mean margin and the
    # smaller ensemble weight are deterministic tie breakers.
    selected = max(
        rows,
        key=lambda row: (
            row["minimum_reference_margin"],
            row["mean_reference_margin"],
            -abs(row["regression_scale"] - 1.0) - abs(row["regression_bias"]),
        ),
    )

    valid_logits, valid_prediction = _blend(
        valid_regression,
        valid_classification,
        selected["classification_weight"],
        selected["regression_scale"],
        selected["regression_bias"],
        selected["classification_head_weight"],
    )
    selected["metrics"] = _paper_metrics(valid, valid_logits, valid_prediction)

    test = load_npz(args.data_root, "test")
    test_regression, _ = _predict(args.regression_checkpoint, test, device)
    test_classification, _ = _predict(args.classification_checkpoint, test, device)
    test_logits, test_prediction = _blend(
        test_regression,
        test_classification,
        selected["classification_weight"],
        selected["regression_scale"],
        selected["regression_bias"],
        selected["classification_head_weight"],
    )
    test_metrics = _paper_metrics(test, test_logits, test_prediction)
    test_margins = _reference_margins(test_metrics, reference)

    result = {
        "method": "validation-selected Pareto checkpoint ensemble",
        "selection_split": "valid",
        "evaluation_split": "test",
        "reference": reference,
        "regression_checkpoint": {
            "path": str(args.regression_checkpoint),
            "epoch": int(regression_payload["epoch"]),
        },
        "classification_checkpoint": {
            "path": str(args.classification_checkpoint),
            "epoch": int(classification_payload["epoch"]),
        },
        "selected": selected,
        "validation_grid_size": len(rows),
        "validation_top_candidates": sorted(
            rows,
            key=lambda row: (
                row["minimum_reference_margin"], row["mean_reference_margin"]
            ),
            reverse=True,
        )[:100],
        "test": {
            "n": int(len(test["regression_labels"])),
            "metrics": test_metrics,
            "reference_margins": test_margins,
            "exceeds_reference_on_all_four": bool(min(test_margins.values()) > 0.0),
        },
    }
    save_json(args.output, result)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "validation_top_candidates"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
