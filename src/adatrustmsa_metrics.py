"""Metrics shared by AdaTrustMSA training, evaluation and policy selection."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error


def consistent_predictions(logits, raw_regression):
    logits = np.asarray(logits)
    raw = np.asarray(raw_regression)
    predicted_class = logits.argmax(axis=1).astype(np.int64)
    magnitude = np.clip(np.abs(raw), 1e-4, 3.0)
    intensity = np.where(
        predicted_class == 1,
        0.0,
        np.where(predicted_class == 0, -magnitude, magnitude),
    )
    return predicted_class, intensity


def softmax(values):
    values = np.asarray(values, dtype=np.float64)
    shifted = values - values.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def expected_calibration_error(labels, probabilities, bins=10):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    prediction = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    result = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for low, high in zip(edges[:-1], edges[1:]):
        chosen = (confidence > low) & (confidence <= high)
        if not chosen.any():
            continue
        accuracy = (prediction[chosen] == labels[chosen]).mean()
        result += chosen.mean() * abs(float(accuracy) - float(confidence[chosen].mean()))
    return float(result)


def metrics(labels, regression_labels, logits, raw_regression) -> Dict[str, float]:
    labels = np.asarray(labels)
    regression_labels = np.asarray(regression_labels)
    probabilities = softmax(logits)
    predicted_class, intensity = consistent_predictions(logits, raw_regression)
    pearson = None
    if np.std(intensity) > 1e-10 and np.std(regression_labels) > 1e-10:
        pearson = float(np.corrcoef(intensity, regression_labels)[0, 1])
    one_hot = np.eye(3, dtype=np.float64)[labels]
    paper_prediction = np.asarray(raw_regression, dtype=np.float64).reshape(-1)
    paper_truth = np.asarray(regression_labels, dtype=np.float64).reshape(-1)
    clipped_prediction = np.clip(paper_prediction, -3.0, 3.0)
    clipped_truth = np.clip(paper_truth, -3.0, 3.0)
    nonzero = paper_truth != 0
    nonzero_truth = paper_truth[nonzero] > 0
    nonzero_prediction = paper_prediction[nonzero] > 0
    haszero_truth = paper_truth >= 0
    haszero_prediction = paper_prediction >= 0
    paper_corr = None
    if np.std(paper_prediction) > 1e-10 and np.std(paper_truth) > 1e-10:
        paper_corr = float(np.corrcoef(paper_prediction, paper_truth)[0, 1])
    result = {
        "n": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predicted_class)),
        "macro_f1": float(
            f1_score(labels, predicted_class, labels=[0, 1, 2], average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(labels, predicted_class, labels=[0, 1, 2], average="weighted", zero_division=0)
        ),
        "mae": float(mean_absolute_error(regression_labels, intensity)),
        "pearson": pearson,
        "ece": expected_calibration_error(labels, probabilities),
        "brier": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "mean_max_probability": float(probabilities.max(axis=1).mean()),
        # Standard MMSA regression protocol. These metrics use the raw sentiment
        # regressor and are kept separate from this project's class-consistent
        # predictions above so paper comparisons cannot silently mix protocols.
        "mmsa_acc_7": float(np.mean(np.round(clipped_prediction) == np.round(clipped_truth))),
        "mmsa_has0_acc_2": float(accuracy_score(haszero_truth, haszero_prediction)),
        "mmsa_has0_f1": float(
            f1_score(haszero_truth, haszero_prediction, average="weighted", zero_division=0)
        ),
        "mmsa_mae": float(mean_absolute_error(paper_truth, paper_prediction)),
        "mmsa_corr": paper_corr,
    }
    if nonzero.any():
        result.update(
            {
                "mmsa_non0_acc_2": float(
                    accuracy_score(nonzero_truth, nonzero_prediction)
                ),
                "mmsa_non0_f1": float(
                    f1_score(
                        nonzero_truth,
                        nonzero_prediction,
                        average="weighted",
                        zero_division=0,
                    )
                ),
            }
        )
    else:
        result.update({"mmsa_non0_acc_2": None, "mmsa_non0_f1": None})
    return result


def selective_curve(labels, predictions, confidence, coverages=(0.5, 0.7, 0.8, 0.9, 1.0)):
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    confidence = np.asarray(confidence)
    order = np.argsort(-confidence)
    rows = []
    for coverage in coverages:
        count = max(1, int(np.ceil(len(labels) * coverage)))
        keep = order[:count]
        rows.append(
            {
                "coverage": float(count / len(labels)),
                "accepted": int(count),
                "accuracy": float((labels[keep] == predictions[keep]).mean()),
                "error_rate": float((labels[keep] != predictions[keep]).mean()),
                "minimum_confidence": float(confidence[keep].min()),
            }
        )
    return rows
