"""Evaluate one checkpoint with the standard MMSA clean regression protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from adatrustmsa_corruption import INPUT_KEYS
from adatrustmsa_metrics import metrics
from adatrustmsa_policy import predict_numpy
from evaluate import load_model
from train import load_npz, save_json


PAPER_KEYS = (
    "mmsa_acc_7",
    "mmsa_non0_acc_2",
    "mmsa_non0_f1",
    "mmsa_has0_acc_2",
    "mmsa_has0_f1",
    "mmsa_mae",
    "mmsa_corr",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", default="test", choices=("valid", "test"))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()

    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    data = load_npz(args.data_root, args.split)
    model, payload = load_model(args.checkpoint, device)
    output = predict_numpy(
        model,
        {key: data[key] for key in INPUT_KEYS},
        device,
        int(payload["config"]["eval_batch_size"]),
    )
    all_metrics = metrics(
        data["classification_labels"],
        data["regression_labels"],
        output["logits"],
        output["regression"],
    )
    result = {
        "protocol": "MMSA clean regression",
        "split": args.split,
        "n": int(len(data["regression_labels"])),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(payload["epoch"]),
        "selected_on": payload.get("validation", {}).get(
            "selection_mode", payload.get("config", {}).get("selection_mode", "validation")
        ),
        **{key: all_metrics[key] for key in PAPER_KEYS},
    }
    save_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
