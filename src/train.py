"""Train AdaTrustMSA on a prepared CMU-MOSEI dataset."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from adatrustmsa_corruption import INPUT_KEYS, corrupt_batch
from adatrustmsa_metrics import metrics
from adatrustmsa_model import AdaTrustMSAModel, load_matching_backbone_weights
from adatrustmsa_policy import predict_numpy


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def load_npz(data_root: Path, split: str) -> dict:
    path = data_root / f"{split}.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run prepare_data.py or point --data-root at prepared data."
        )
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    for key in (*INPUT_KEYS, "classification_labels", "regression_labels"):
        if key not in data:
            raise KeyError(f"{path} does not contain required array {key!r}")
    return data


def batch_inputs(data: dict, indices: np.ndarray) -> dict:
    return {key: data[key][indices] for key in INPUT_KEYS}


def tensor_inputs(data: dict, device) -> dict:
    return {key: torch.as_tensor(data[key], device=device) for key in INPUT_KEYS}


def make_model_kwargs(encoder: Path, preprocessing: Path, config: dict, ablation: str) -> dict:
    if not (encoder / "config.json").is_file():
        raise FileNotFoundError(f"BERT config not found in {encoder}")
    if not preprocessing.is_file():
        raise FileNotFoundError(f"Preprocessing statistics not found: {preprocessing}")
    flags = {
        "full": (True, True, True),
        "no_reliability": (False, True, True),
        "no_conflict": (True, False, True),
        "no_uncertainty": (True, True, False),
    }
    if ablation not in flags:
        raise ValueError(f"unknown ablation: {ablation}")
    use_reliability, use_conflict, use_uncertainty = flags[ablation]
    return {
        "bert_config": json.loads((encoder / "config.json").read_text(encoding="utf-8")),
        **json.loads(preprocessing.read_text(encoding="utf-8")),
        "hidden": int(config["hidden"]),
        "dropout": float(config["dropout"]),
        "uncertainty_penalty": float(config["uncertainty_penalty"]),
        "conflict_penalty": float(config["conflict_penalty"]),
        "fusion_layers": int(config.get("fusion_layers", 2)),
        "fusion_heads": int(config.get("fusion_heads", 4)),
        "fusion_expansion": int(config.get("fusion_expansion", 2)),
        "prediction_blending": bool(config.get("prediction_blending", False)),
        "attentive_pooling": bool(config.get("attentive_pooling", False)),
        "quality_attention": bool(config.get("quality_attention", False)),
        "quality_attention_strength": float(
            config.get("quality_attention_strength", 1.0)
        ),
        "use_reliability": use_reliability,
        "use_conflict": use_conflict,
        "use_uncertainty": use_uncertainty,
    }


def compute_loss(
    output,
    labels,
    regression,
    reliability_target,
    class_weights,
    weights,
    label_smoothing=0.0,
    distillation_temperature=2.0,
    regression_boundary_temperature=0.5,
):
    classification = F.cross_entropy(
        output["logits"].float(),
        labels,
        weight=class_weights,
        label_smoothing=float(label_smoothing),
    )
    residual = regression - output["regression"].float()
    heteroscedastic = 0.5 * (
        torch.exp(-output["log_variance"].float()) * residual.square()
        + output["log_variance"].float()
    ).mean()
    reliability = F.mse_loss(output["reliability"].float(), reliability_target)
    fused_regression = F.smooth_l1_loss(
        output["regression"].float(),
        regression,
        beta=0.5,
    )
    raw_regression = output["regression"].float()
    nonzero = regression != 0
    if nonzero.any():
        sign_target = torch.where(
            regression[nonzero] > 0,
            torch.ones_like(regression[nonzero]),
            -torch.ones_like(regression[nonzero]),
        )
        sign_boundary = F.softplus(
            -sign_target
            * raw_regression[nonzero]
            / float(regression_boundary_temperature)
        ).mean()
    else:
        sign_boundary = raw_regression.sum() * 0.0

    # Six ordered half-integer boundaries are exactly the decision boundaries
    # used by the standard CMU-MOSEI Acc-7 protocol after clipping/rounding.
    ordinal_thresholds = raw_regression.new_tensor(
        [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    )
    ordinal_logits = (
        raw_regression[:, None] - ordinal_thresholds[None, :]
    ) / float(regression_boundary_temperature)
    ordinal_targets = (
        regression[:, None] > ordinal_thresholds[None, :]
    ).to(ordinal_logits.dtype)
    ordinal_regression = F.binary_cross_entropy_with_logits(
        ordinal_logits, ordinal_targets
    )

    prediction_centered = raw_regression - raw_regression.mean()
    target_centered = regression - regression.mean()
    correlation = (
        (prediction_centered * target_centered).sum()
        / (
            prediction_centered.square().sum().sqrt()
            * target_centered.square().sum().sqrt()
        ).clamp_min(1e-6)
    )
    correlation_regression = 1.0 - correlation

    repeated_labels = labels[:, None].expand(-1, 3).reshape(-1)
    unimodal_ce = F.cross_entropy(
        output["modality_logits"].float().reshape(-1, 3),
        repeated_labels,
        reduction="none",
    ).view(-1, 3)
    quality_weight = reliability_target.detach()
    denominator = quality_weight.sum().clamp_min(1.0)
    unimodal_classification = (unimodal_ce * quality_weight).sum() / denominator
    unimodal_regression = F.smooth_l1_loss(
        output["modality_regression"].float(),
        regression[:, None].expand(-1, 3),
        reduction="none",
        beta=0.5,
    )
    unimodal_regression = (unimodal_regression * quality_weight).sum() / denominator

    repeated_expert_labels = labels[:, None].expand(-1, 4).reshape(-1)
    expert_ce = F.cross_entropy(
        output["expert_logits"].float().reshape(-1, 3),
        repeated_expert_labels,
        weight=class_weights,
        reduction="none",
        label_smoothing=float(label_smoothing),
    ).view(-1, 4)
    expert_target = torch.cat(
        (
            reliability_target.detach(),
            reliability_target.detach().mean(dim=1, keepdim=True),
        ),
        dim=1,
    ) * output["expert_present"].detach().to(torch.float32)
    expert_classification = (expert_ce * expert_target).sum() / expert_target.sum().clamp_min(1.0)

    probabilities = torch.softmax(output["logits"].float(), dim=-1)
    one_hot = F.one_hot(labels, num_classes=3).float()
    brier = (probabilities - one_hot).square().sum(dim=-1).mean()
    temperature = float(distillation_temperature)
    with torch.no_grad():
        modality_probability = torch.softmax(
            output["modality_logits"].float() / temperature,
            dim=-1,
        )
        modality_weight = output["modality_weights"].float().unsqueeze(-1)
        unimodal_teacher = (modality_probability * modality_weight).sum(dim=1)
        unimodal_teacher = unimodal_teacher / unimodal_teacher.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
    unimodal_distillation = F.kl_div(
        F.log_softmax(output["logits"].float() / temperature, dim=-1),
        unimodal_teacher,
        reduction="batchmean",
    ) * temperature**2
    terms = {
        "classification": classification,
        "heteroscedastic_regression": heteroscedastic,
        "fused_regression": fused_regression,
        "sign_boundary": sign_boundary,
        "ordinal_regression": ordinal_regression,
        "correlation_regression": correlation_regression,
        "reliability": reliability,
        "unimodal_classification": unimodal_classification,
        "unimodal_regression": unimodal_regression,
        "expert_classification": expert_classification,
        "unimodal_distillation": unimodal_distillation,
        "brier": brier,
    }
    total = sum(float(weights.get(name, 0.0)) * value for name, value in terms.items())
    return total, terms


def evaluate(model, valid: dict, device, config: dict) -> dict:
    clean_inputs = {key: valid[key] for key in INPUT_KEYS}
    clean_output = predict_numpy(
        model, clean_inputs, device, int(config["eval_batch_size"])
    )
    clean = metrics(
        valid["classification_labels"],
        valid["regression_labels"],
        clean_output["logits"],
        clean_output["regression"],
    )
    validation = config["validation"]
    rng = np.random.default_rng(int(validation["corruption_seed"]))
    ratio = float(validation["corruption_ratio"])
    corrupted, target, audit = corrupt_batch(
        clean_inputs,
        rng,
        probability=1.0,
        ratio_range=(ratio, ratio),
        gaussian_probability=float(config["corruption"]["gaussian_probability"]),
        audio_std=model.audio_std.detach().cpu().numpy(),
        vision_std=model.vision_std.detach().cpu().numpy(),
    )
    robust_output = predict_numpy(
        model, corrupted, device, int(config["eval_batch_size"])
    )
    robust = metrics(
        valid["classification_labels"],
        valid["regression_labels"],
        robust_output["logits"],
        robust_output["regression"],
    )
    balanced_mae = 0.5 * (clean["mae"] + robust["mae"])
    selection_mode = validation.get("selection_mode", "project_robust")
    if selection_mode == "mmsa_clean_mae":
        # Standard clean-benchmark runs select only on validation regression
        # MAE. The test labels remain untouched until the final evaluation.
        selection_score = -clean["mmsa_mae"]
    elif selection_mode == "mmsa_pareto":
        corr = clean["mmsa_corr"] if clean["mmsa_corr"] is not None else -1.0
        selection_score = (
            0.15 * clean["mmsa_acc_7"]
            + 0.30 * clean["mmsa_non0_acc_2"]
            + 0.30 * clean["mmsa_non0_f1"]
            + 0.15 * (1.0 - clean["mmsa_mae"] / 3.0)
            + 0.10 * ((corr + 1.0) / 2.0)
        )
    elif selection_mode == "project_robust":
        selection_score = (
            0.35 * clean["macro_f1"]
            + 0.35 * robust["macro_f1"]
            + 0.15 * (1.0 - clean["ece"])
            + 0.15 * (1.0 - balanced_mae / 6.0)
        )
    else:
        raise ValueError(f"unknown validation selection_mode: {selection_mode}")
    corruption_summary = {
        "samples": len(audit),
        "mean_reliability_target": target.mean(axis=0).tolist(),
        "missing_count": int(sum(item.get("kind") == "missing" for item in audit)),
        "gaussian_count": int(sum(item.get("kind") == "gaussian" for item in audit)),
    }
    return {
        "selection_score": float(selection_score),
        "selection_mode": selection_mode,
        "balanced_mae": float(balanced_mae),
        "clean": clean,
        "robust": robust,
        "corruption": corruption_summary,
    }


def checkpoint_payload(model, model_kwargs, config, epoch, validation, warm_start):
    return {
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "model_kwargs": model_kwargs,
        "config": config,
        "epoch": int(epoch),
        "validation": validation,
        "warm_start": warm_start,
        "format": "adatrustmsa_1.0_complete_checkpoint",
    }


def train(args) -> None:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed = int(args.seed if args.seed is not None else config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    train_data = load_npz(args.data_root, "train")
    valid_data = load_npz(args.data_root, "valid")
    preprocessing = args.preprocessing or (args.data_root / "preprocessing.json")
    model_kwargs = make_model_kwargs(args.encoder, preprocessing, config, args.ablation)
    if args.warm_start and args.initialize_from:
        raise ValueError("--warm-start and --initialize-from are mutually exclusive")
    pretrained = None if (args.warm_start or args.initialize_from) else str(args.encoder)
    model = AdaTrustMSAModel(**model_kwargs, bert_pretrained_dir=pretrained)
    warm_start = None
    if args.warm_start:
        warm_start = load_matching_backbone_weights(model, str(args.warm_start))
    elif args.initialize_from:
        source = torch.load(args.initialize_from, map_location="cpu", weights_only=True)
        model.load_state_dict(source["state_dict"], strict=True)
        warm_start = {
            "source": str(args.initialize_from),
            "mode": "complete_model_finetune",
            "source_epoch": int(source.get("epoch", -1)),
            "matched_parameter_count": int(
                sum(parameter.numel() for parameter in model.parameters())
            ),
        }
    model.to(device)

    counts = np.bincount(train_data["classification_labels"], minlength=3)
    class_weight_power = float(config.get("class_weight_power", 0.5))
    raw_weights = (
        len(train_data["classification_labels"]) / np.maximum(3 * counts, 1)
    ) ** class_weight_power
    raw_weights /= raw_weights.mean()
    class_weights = torch.as_tensor(raw_weights, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.bert.parameters(), "lr": float(config["bert_lr"])},
            {
                "params": [parameter for name, parameter in model.named_parameters() if not name.startswith("bert.")],
                "lr": float(config["head_lr"]),
            },
        ],
        weight_decay=float(config["weight_decay"]),
    )
    batch_size = int(config["batch_size"])
    epochs = int(args.epochs if args.epochs is not None else config["epochs"])
    steps = epochs * math.ceil(len(train_data["input_ids"]) / batch_size)
    warmup = max(1, int(steps * float(config["warmup_fraction"])))

    def lr_factor(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, steps - warmup)
        return 0.1 + 0.9 * (1 + math.cos(math.pi * min(1.0, progress))) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    amp = device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
    else:  # PyTorch 2.0/2.1 compatibility for existing CUDA server images.
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        args.output_dir / "resolved_config.json",
        {**config, "seed": seed, "device": str(device), "ablation": args.ablation},
    )
    if warm_start is not None:
        save_json(args.output_dir / "warm_start_audit.json", warm_start)

    best_score = -float("inf")
    best_epoch = 0
    history = []
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        freeze_epochs = int(config.get("bert_freeze_epochs", 0))
        top_layers = int(config.get("bert_unfreeze_top_layers", 0))
        if epoch <= freeze_epochs:
            for parameter in model.bert.parameters():
                parameter.requires_grad_(False)
            model.bert.eval()
        elif top_layers > 0:
            for parameter in model.bert.parameters():
                parameter.requires_grad_(False)
            encoder_layers = model.bert.encoder.layer
            for layer in encoder_layers[-top_layers:]:
                for parameter in layer.parameters():
                    parameter.requires_grad_(True)
        else:
            for parameter in model.bert.parameters():
                parameter.requires_grad_(True)
        rng = np.random.default_rng(seed + epoch * 1009)
        order = rng.permutation(len(train_data["input_ids"]))
        totals = {"total": 0.0, **{name: 0.0 for name in config["loss_weights"]}}
        seen = 0
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            numpy_inputs = batch_inputs(train_data, indices)
            corrupted, reliability_target, _ = corrupt_batch(
                numpy_inputs,
                rng,
                probability=float(config["corruption"]["probability"]),
                ratio_range=tuple(config["corruption"]["ratio_range"]),
                gaussian_probability=float(config["corruption"]["gaussian_probability"]),
                cascade_dropout_probability=float(
                    config["corruption"]["cascade_dropout_probability"]
                ),
                audio_std=np.asarray(model_kwargs["audio_std"], np.float32),
                vision_std=np.asarray(model_kwargs["vision_std"], np.float32),
            )
            inputs = tensor_inputs(corrupted, device)
            labels = torch.as_tensor(train_data["classification_labels"][indices], device=device, dtype=torch.long)
            regression = torch.as_tensor(train_data["regression_labels"][indices], device=device, dtype=torch.float32)
            target = torch.as_tensor(reliability_target, device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            context = torch.autocast("cuda", dtype=torch.float16) if amp else nullcontext()
            with context:
                output = model(**inputs)
                loss, terms = compute_loss(
                    output,
                    labels,
                    regression,
                    target,
                    class_weights,
                    config["loss_weights"],
                    config.get("label_smoothing", 0.0),
                    config.get("distillation_temperature", 2.0),
                    config.get("regression_boundary_temperature", 0.5),
                )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            size = len(indices)
            totals["total"] += float(loss.detach()) * size
            for name, value in terms.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * size
            seen += size

        validation = evaluate(model, valid_data, device, config)
        improved = validation["selection_score"] > best_score + 1e-7
        if improved:
            best_score = validation["selection_score"]
            best_epoch = epoch
            torch.save(
                checkpoint_payload(model, model_kwargs, config, epoch, validation, warm_start),
                args.output_dir / "best.pt",
            )
            save_json(args.output_dir / "best_validation.json", validation)
        record = {
            "epoch": epoch,
            "seconds": float(time.monotonic() - started),
            "loss": {name: value / seen for name, value in totals.items()},
            "selection_score": validation["selection_score"],
            "clean_macro_f1": validation["clean"]["macro_f1"],
            "robust_macro_f1": validation["robust"]["macro_f1"],
            "clean_mae": validation["clean"]["mae"],
            "robust_mae": validation["robust"]["mae"],
            "clean_ece": validation["clean"]["ece"],
            "clean_mmsa_acc_7": validation["clean"]["mmsa_acc_7"],
            "clean_mmsa_non0_acc_2": validation["clean"]["mmsa_non0_acc_2"],
            "clean_mmsa_non0_f1": validation["clean"]["mmsa_non0_f1"],
            "clean_mmsa_mae": validation["clean"]["mmsa_mae"],
            "clean_mmsa_corr": validation["clean"]["mmsa_corr"],
            "bert_trainable_parameters": int(
                sum(
                    parameter.numel()
                    for parameter in model.bert.parameters()
                    if parameter.requires_grad
                )
            ),
            "improved": improved,
            "best_epoch": best_epoch,
        }
        history.append(record)
        save_json(args.output_dir / "history.json", history)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if epoch - best_epoch >= int(config["early_stopping_patience"]):
            break

    save_json(
        args.output_dir / "summary.json",
        {
            "status": "complete",
            "best_epoch": best_epoch,
            "best_selection_score": best_score,
            "epochs_run": len(history),
            "elapsed_seconds": float(time.monotonic() - started),
            "checkpoint": str(args.output_dir / "best.pt"),
            "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        },
    )


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Folder containing train.npz and valid.npz")
    parser.add_argument("--encoder", type=Path, required=True, help="Local Hugging Face BERT folder")
    parser.add_argument(
        "--preprocessing",
        type=Path,
        help="Normalization JSON; defaults to DATA_ROOT/preprocessing.json",
    )
    parser.add_argument("--config", type=Path, default=project / "configs/adatrustmsa.json")
    parser.add_argument("--output-dir", type=Path, default=project / "runs/adatrustmsa")
    parser.add_argument("--warm-start", type=Path, help="Optional compatible backbone checkpoint")
    parser.add_argument(
        "--initialize-from",
        type=Path,
        help="Fine-tune a complete checkpoint with an identical model architecture",
    )
    parser.add_argument(
        "--ablation",
        choices=("full", "no_reliability", "no_conflict", "no_uncertainty"),
        default="full",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
