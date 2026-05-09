#!/usr/bin/env python3
"""Train and evaluate a CNN+LSTM clip classifier."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..cnn_baseline.pipeline import BaselineCNN
from ..cnn_advanced.pipeline import AdvancedCNN
from ..resnet18.pipeline import ResNet18PunchClassifier
from ..vgg_16.pipeline import VGG16PunchClassifier
from .utils import (
    ClipDataset,
    compute_class_weights,
    evaluate_and_save,
    get_data_config,
    get_transforms,
    load_checkpoint,
    load_manifest,
    parse_class_names,
    resolve_device,
    resolve_latest_checkpoint,
    resolve_project_path,
    run_epoch,
    save_checkpoint,
    save_json,
    set_seed,
    split_by_video,
)


class CNNFeatureExtractor(nn.Module):
    def __init__(self, backbone: str, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "cnn_baseline":
            self.model = BaselineCNN(num_classes=num_classes)
            self.feature_dim = 256
        elif backbone == "cnn_advanced":
            self.model = AdvancedCNN(num_classes=num_classes)
            self.feature_dim = 1024
        elif backbone == "resnet18":
            self.model = ResNet18PunchClassifier(num_classes=num_classes, pretrained=pretrained)
            self.feature_dim = 512
        elif backbone == "vgg_16":
            self.model = VGG16PunchClassifier(num_classes=num_classes, pretrained=pretrained)
            self.feature_dim = 25088
        else:
            raise SystemExit(
                "Unknown cnn_backbone. Use one of: cnn_baseline, cnn_advanced, resnet18, vgg_16"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone_name == "cnn_baseline":
            feats = self.model.features(x)
            return feats.flatten(1)
        if self.backbone_name == "cnn_advanced":
            x = self.model.block1(x)
            x = self.model.res1(x)
            x = self.model.res2(x)
            x = self.model.res3(x)
            att_weights = self.model.attention(x)
            att_weights = att_weights.view(x.size(0), 512, 1, 1)
            x = x * att_weights
            x = self.model.features(x)
            return x.flatten(1)
        if self.backbone_name == "resnet18":
            resnet = self.model.resnet
            x = resnet.conv1(x)
            x = resnet.bn1(x)
            x = resnet.relu(x)
            x = resnet.maxpool(x)
            x = resnet.layer1(x)
            x = resnet.layer2(x)
            x = resnet.layer3(x)
            x = resnet.layer4(x)
            x = resnet.avgpool(x)
            return torch.flatten(x, 1)
        if self.backbone_name == "vgg_16":
            x = self.model.vgg16.features(x)
            x = self.model.vgg16.avgpool(x)
            return torch.flatten(x, 1)
        raise SystemExit("Unsupported backbone")


class CNNLSTMClassifier(nn.Module):
    def __init__(
        self,
        backbone: CNNFeatureExtractor,
        num_classes: int,
        lstm_hidden: int,
        lstm_layers: int,
        lstm_dropout: float,
        lstm_bidirectional: bool,
        lstm_output: str,
        projection_dim: int | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.lstm_output = lstm_output

        feature_dim = backbone.feature_dim
        if projection_dim and projection_dim > 0 and projection_dim != feature_dim:
            self.proj = nn.Linear(feature_dim, projection_dim)
            feature_dim = projection_dim
        else:
            self.proj = None

        dropout = lstm_dropout if lstm_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=lstm_bidirectional,
        )
        lstm_out_dim = lstm_hidden * (2 if lstm_bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(lstm_out_dim, num_classes),
        )

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = clips.shape
        flat = clips.view(b * t, c, h, w)
        feats = self.backbone(flat)
        feats = feats.view(b, t, -1)

        if self.proj is not None:
            feats = self.proj(feats)

        lstm_out, _ = self.lstm(feats)
        if self.lstm_output == "mean":
            pooled = lstm_out.mean(dim=1)
        else:
            pooled = lstm_out[:, -1, :]
        return self.classifier(pooled)


def _resolve_cnn_init(init_value: str, backbone: str) -> tuple[bool, Path | None]:
    init_key = (init_value or "").strip().lower()
    if init_key == "latest":
        return False, resolve_latest_checkpoint(backbone)
    if init_key in {"imagenet", "pretrained"}:
        return True, None
    if init_key in {"random", "none"}:
        return False, None
    return False, resolve_project_path(init_value)


def _maybe_load_checkpoint(model: nn.Module, checkpoint_path: Path | None, backbone: str, device: torch.device) -> None:
    if checkpoint_path is None:
        return
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device)
    state = payload.get("model_state_dict", payload)
    model.load_state_dict(state, strict=False)


def train(args: argparse.Namespace) -> None:
    class_names = parse_class_names(args.class_names)

    set_seed(args.seed)
    if args.epochs < 1:
        raise SystemExit("--epochs must be >= 1")
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    config = get_data_config()
    data_cfg = config.get("data", {})
    manifest_path = resolve_project_path(
        getattr(args, "manifest_file", "") or data_cfg.get("clip_manifest_file", "data/processed/clip_manifest.csv")
    )
    df = load_manifest(manifest_path)

    splits = getattr(args, "splits", None)
    if not isinstance(splits, dict):
        raise ValueError("The 'splits' configuration is missing from the hparams file.")
    train_df, val_df, test_df = split_by_video(df, splits, args.seed)

    augment = getattr(args, "augment", None)
    if augment is None:
        raise ValueError("The 'augment' configuration is missing from the hparams file.")
    train_tf = get_transforms(args.image_size, augment=bool(augment))
    eval_tf = get_transforms(args.image_size, augment=False)

    train_ds = ClipDataset(train_df, clip_length=args.clip_length, transform=train_tf)
    val_ds = ClipDataset(val_df, clip_length=args.clip_length, transform=eval_tf)
    test_ds = ClipDataset(test_df, clip_length=args.clip_length, transform=eval_tf)

    pin_memory = not torch.backends.mps.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    cnn_pretrained, cnn_ckpt = _resolve_cnn_init(args.cnn_init, args.cnn_backbone)
    if args.cnn_init.strip().lower() == "latest" and cnn_ckpt is None:
        print(f"No checkpoint found for backbone '{args.cnn_backbone}'. Falling back to random init.")

    backbone = CNNFeatureExtractor(
        backbone=args.cnn_backbone,
        num_classes=len(class_names),
        pretrained=cnn_pretrained,
    )
    _maybe_load_checkpoint(backbone.model, cnn_ckpt, args.cnn_backbone, device)

    freeze_backbone = getattr(args, "freeze_backbone", None)
    if freeze_backbone is None:
        raise ValueError("The 'freeze_backbone' configuration is missing from the hparams file.")
    if freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False

    model = CNNLSTMClassifier(
        backbone=backbone,
        num_classes=len(class_names),
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        lstm_dropout=args.lstm_dropout,
        lstm_bidirectional=args.lstm_bidirectional,
        lstm_output=args.lstm_output,
        projection_dim=getattr(args, "feature_dim", None),
    ).to(device)

    labels = train_df["label"].to_numpy(dtype=np.int64)
    class_weights = (
        compute_class_weights(labels, len(class_names), device) if args.use_class_weights else None
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(1, args.lr_patience),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = resolve_project_path(args.output_dir)
    run_dir = output_root / f"{args.run_name}_{timestamp}"
    checkpoint_dir = run_dir / "checkpoints"
    report_dir = run_dir / "reports"
    run_dir.mkdir(parents=True, exist_ok=True)

    config_payload = {
        "class_names": class_names,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "image_size": args.image_size,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "device": str(device),
        "use_class_weights": bool(args.use_class_weights),
        "cnn_backbone": args.cnn_backbone,
        "cnn_init": args.cnn_init,
        "freeze_backbone": bool(freeze_backbone),
        "clip_length": args.clip_length,
        "lstm_hidden": args.lstm_hidden,
        "lstm_layers": args.lstm_layers,
        "lstm_dropout": args.lstm_dropout,
        "lstm_bidirectional": bool(args.lstm_bidirectional),
        "lstm_output": args.lstm_output,
    }
    save_json(run_dir / "config.json", config_payload)

    best_macro_f1 = -1.0
    best_epoch = -1
    no_improve_epochs = 0
    history_rows: list[dict] = []

    print(
        f"Train samples: {len(train_df)} | Val samples: {len(val_df)} | "
        f"Test samples: {len(test_df)}"
    )

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_metrics, _, _ = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )
        val_metrics, _, _ = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )

        scheduler.step(val_metrics.macro_f1)
        current_lr = float(optimizer.param_groups[0]["lr"])
        history_rows.append(
            {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": train_metrics.loss,
                "train_acc": train_metrics.accuracy,
                "train_macro_f1": train_metrics.macro_f1,
                "val_loss": val_metrics.loss,
                "val_acc": val_metrics.accuracy,
                "val_macro_f1": val_metrics.macro_f1,
            }
        )

        print(
            "train_loss={:.4f} train_acc={:.4f} train_f1={:.4f} | "
            "val_loss={:.4f} val_acc={:.4f} val_f1={:.4f}".format(
                train_metrics.loss,
                train_metrics.accuracy,
                train_metrics.macro_f1,
                val_metrics.loss,
                val_metrics.accuracy,
                val_metrics.macro_f1,
            )
        )

        save_checkpoint(
            path=checkpoint_dir / "last.pt",
            model=model,
            class_names=class_names,
            image_size=args.image_size,
            epoch=epoch,
            best_val_macro_f1=max(best_macro_f1, val_metrics.macro_f1),
            cnn_backbone=args.cnn_backbone,
            clip_length=args.clip_length,
        )

        if val_metrics.macro_f1 > best_macro_f1:
            best_macro_f1 = val_metrics.macro_f1
            best_epoch = epoch
            no_improve_epochs = 0
            save_checkpoint(
                path=checkpoint_dir / "best.pt",
                model=model,
                class_names=class_names,
                image_size=args.image_size,
                epoch=epoch,
                best_val_macro_f1=best_macro_f1,
                cnn_backbone=args.cnn_backbone,
                clip_length=args.clip_length,
            )
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= args.patience:
            print(f"Early stopping triggered after {args.patience} epochs without val macro-F1 improvement.")
            break

    history_path = run_dir / "history.csv"
    if history_rows:
        with history_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history_rows[0].keys()))
            writer.writeheader()
            writer.writerows(history_rows)

    print(f"\nBest validation macro-F1: {best_macro_f1:.4f} at epoch {best_epoch}")

    best_ckpt = load_checkpoint(checkpoint_dir / "best.pt", device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    if len(val_loader.dataset) == 0:
        print("Validation summary: skipped (no val samples)")
    else:
        val_summary = evaluate_and_save(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            class_names=class_names,
            out_dir=report_dir,
            split_name="val",
        )
        print("Validation summary:", json.dumps(val_summary, indent=2))

    if len(test_loader.dataset) == 0:
        print("Test summary: skipped (no test samples)")
    else:
        test_summary = evaluate_and_save(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            class_names=class_names,
            out_dir=report_dir,
            split_name="test",
        )
        print("Test summary:", json.dumps(test_summary, indent=2))

    summary = {
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro_f1,
        "checkpoint_best": str(checkpoint_dir / "best.pt"),
        "report_dir": str(report_dir),
    }
    save_json(run_dir / "run_summary.json", summary)
    print(f"\nRun complete. Artifacts saved to: {run_dir}")


def _load_for_eval(args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, list[str]]:
    ckpt_path = resolve_project_path(args.checkpoint)
    checkpoint = load_checkpoint(ckpt_path, device)
    class_names = checkpoint.get("class_names")
    if not class_names:
        class_names = parse_class_names(args.class_names)

    cnn_pretrained, _ = _resolve_cnn_init(args.cnn_init, args.cnn_backbone)
    backbone = CNNFeatureExtractor(
        backbone=args.cnn_backbone,
        num_classes=len(class_names),
        pretrained=cnn_pretrained,
    )
    model = CNNLSTMClassifier(
        backbone=backbone,
        num_classes=len(class_names),
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        lstm_dropout=args.lstm_dropout,
        lstm_bidirectional=args.lstm_bidirectional,
        lstm_output=args.lstm_output,
        projection_dim=getattr(args, "feature_dim", None),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names


def test_only(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model, class_names = _load_for_eval(args, device)

    config = get_data_config()
    data_cfg = config.get("data", {})
    manifest_path = resolve_project_path(
        getattr(args, "manifest_file", "") or data_cfg.get("clip_manifest_file", "data/processed/clip_manifest.csv")
    )
    df = load_manifest(manifest_path)
    splits = getattr(args, "splits", None)
    if not isinstance(splits, dict):
        raise ValueError("The 'splits' configuration is missing from the hparams file.")
    _, _, test_df = split_by_video(df, splits, args.seed)

    eval_tf = get_transforms(args.image_size, augment=False)
    test_ds = ClipDataset(test_df, clip_length=args.clip_length, transform=eval_tf)
    pin_memory = not torch.backends.mps.is_available()
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    criterion = nn.CrossEntropyLoss()
    report_dir = resolve_project_path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    test_summary = evaluate_and_save(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        class_names=class_names,
        out_dir=report_dir,
        split_name="test",
    )
    print("Test summary:", json.dumps(test_summary, indent=2))


def validate_only(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model, class_names = _load_for_eval(args, device)

    config = get_data_config()
    data_cfg = config.get("data", {})
    manifest_path = resolve_project_path(
        getattr(args, "manifest_file", "") or data_cfg.get("clip_manifest_file", "data/processed/clip_manifest.csv")
    )
    df = load_manifest(manifest_path)
    splits = getattr(args, "splits", None)
    if not isinstance(splits, dict):
        raise ValueError("The 'splits' configuration is missing from the hparams file.")
    _, val_df, _ = split_by_video(df, splits, args.seed)

    eval_tf = get_transforms(args.image_size, augment=False)
    val_ds = ClipDataset(val_df, clip_length=args.clip_length, transform=eval_tf)
    pin_memory = not torch.backends.mps.is_available()
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    criterion = nn.CrossEntropyLoss()
    report_dir = resolve_project_path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    val_summary = evaluate_and_save(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        class_names=class_names,
        out_dir=report_dir,
        split_name="val",
    )
    print("Validation summary:", json.dumps(val_summary, indent=2))
