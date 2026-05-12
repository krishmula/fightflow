#!/usr/bin/env python3
"""Train and evaluate a Pose-based LSTM classifier."""

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
from tqdm import tqdm

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque

from .utils import (
    POSE_CONNECTIONS,
    PoseDataset,
    compute_class_weights,
    evaluate_and_save,
    get_data_config,
    load_checkpoint,
    load_manifest,
    parse_class_names,
    resolve_device,
    resolve_project_path,
    run_epoch,
    save_checkpoint,
    save_json,
    set_seed,
    split_by_video,
)

class PoseLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        lstm_hidden: int,
        lstm_layers: int,
        lstm_dropout: float,
        lstm_bidirectional: bool,
        classifier_dropout: float = 0.5,
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=lstm_bidirectional,
        )
        
        lstm_out_dim = lstm_hidden * (2 if lstm_bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=classifier_dropout),
            nn.Linear(lstm_out_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, features)
        lstm_out, _ = self.lstm(x)
        # Global average pooling over time or last state
        pooled = lstm_out.mean(dim=1)
        return self.classifier(pooled)

def train(args: argparse.Namespace) -> None:
    class_names = parse_class_names(args.class_names)
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    config = get_data_config()
    data_cfg = config.get("data", {})
    manifest_path = resolve_project_path(
        getattr(args, "manifest_file", "") or data_cfg.get("pose_manifest_file", "data/processed/pose_manifest.csv")
    )
    df = load_manifest(manifest_path)
    train_df, val_df, test_df = split_by_video(df, args.splits, args.seed)

    pose_root = resolve_project_path(args.pose_root)
    
    # Enable noise augmentation only for training
    noise_std = getattr(args, "pose_noise_std", 0.0)
    train_ds = PoseDataset(train_df, pose_root, clip_length=args.clip_length, noise_std=noise_std)
    val_ds = PoseDataset(val_df, pose_root, clip_length=args.clip_length)
    test_ds = PoseDataset(test_df, pose_root, clip_length=args.clip_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 33 landmarks * 3 coordinates (x, y, z) = 99
    input_dim = 99 
    
    model = PoseLSTMClassifier(
        input_dim=input_dim,
        num_classes=len(class_names),
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        lstm_dropout=args.lstm_dropout,
        lstm_bidirectional=args.lstm_bidirectional,
        classifier_dropout=args.classifier_dropout,
    ).to(device)

    # Optional: Resume from checkpoint
    resume_path = getattr(args, "resume", None)
    if resume_path:
        print(f"Resuming training from: {resume_path}")
        checkpoint = load_checkpoint(resolve_project_path(resume_path), device)
        model.load_state_dict(checkpoint["model_state_dict"])

    # Label smoothing to prevent over-confidence and improve generalization
    label_smoothing = getattr(args, "label_smoothing", 0.0)
    class_weights = compute_class_weights(train_df["label"].values, len(class_names), device) if args.use_class_weights else None
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=args.lr_patience)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = resolve_project_path(args.output_dir) / f"{args.run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    report_dir = run_dir / "reports"

    best_macro_f1 = -1.0
    best_epoch = -1
    no_improve_epochs = 0
    history_rows = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics, _, _ = run_epoch(train_loader, model, criterion, device, optimizer)
        val_metrics, _, _ = run_epoch(val_loader, model, criterion, device)

        scheduler.step(val_metrics.macro_f1)
        
        # Epoch summary matching cnn_lstm
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
            epoch=epoch,
            best_val_macro_f1=max(best_macro_f1, val_metrics.macro_f1),
        )

        if val_metrics.macro_f1 > best_macro_f1:
            best_macro_f1 = val_metrics.macro_f1
            best_epoch = epoch
            no_improve_epochs = 0
            save_checkpoint(
                path=checkpoint_dir / "best.pt",
                model=model,
                epoch=epoch,
                best_val_macro_f1=best_macro_f1,
            )
        else:
            no_improve_epochs += 1

        history_rows.append({
            "epoch": epoch,
            "train_loss": train_metrics.loss,
            "train_acc": train_metrics.accuracy,
            "train_f1": train_metrics.macro_f1,
            "val_loss": val_metrics.loss,
            "val_acc": val_metrics.accuracy,
            "val_f1": val_metrics.macro_f1,
        })

        if no_improve_epochs >= args.patience:
            print(f"Early stopping triggered after {args.patience} epochs.")
            break

    # Save history matching cnn_lstm
    history_path = run_dir / "history.csv"
    if history_rows:
        with history_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history_rows[0].keys()))
            writer.writeheader()
            writer.writerows(history_rows)

    print(f"\nBest validation macro-F1: {best_macro_f1:.4f} at epoch {best_epoch}")

    # Load best and evaluate on test
    best_ckpt = load_checkpoint(checkpoint_dir / "best.pt", device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    
    test_summary = evaluate_and_save(model, test_loader, criterion, device, class_names, report_dir, "test")
    
    return {
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro_f1,
        "checkpoint_best": str(checkpoint_dir / "best.pt"),
        "report_dir": str(report_dir),
    }

def test_only(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    checkpoint_path = resolve_project_path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, device)
    class_names = parse_class_names(args.class_names)
    
    pose_root = resolve_project_path(args.pose_root)
    config = get_data_config()
    data_cfg = config.get("data", {})
    manifest_path = resolve_project_path(
        getattr(args, "manifest_file", "") or data_cfg.get("pose_manifest_file", "data/processed/pose_manifest.csv")
    )
    _, _, test_df = split_by_video(load_manifest(manifest_path), args.splits, args.seed)
    test_ds = PoseDataset(test_df, pose_root, clip_length=args.clip_length)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = PoseLSTMClassifier(
        input_dim=99,
        num_classes=len(class_names),
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        lstm_dropout=args.lstm_dropout,
        lstm_bidirectional=args.lstm_bidirectional,
        classifier_dropout=args.classifier_dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    criterion = nn.CrossEntropyLoss()
    report_dir = resolve_project_path(args.report_dir)
    evaluate_and_save(model, test_loader, criterion, device, class_names, report_dir, "test")

def validate_only(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    checkpoint_path = resolve_project_path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, device)
    class_names = parse_class_names(args.class_names)
    
    pose_root = resolve_project_path(args.pose_root)
    config = get_data_config()
    data_cfg = config.get("data", {})
    manifest_path = resolve_project_path(
        getattr(args, "manifest_file", "") or data_cfg.get("pose_manifest_file", "data/processed/pose_manifest.csv")
    )
    _, val_df, _ = split_by_video(load_manifest(manifest_path), args.splits, args.seed)
    val_ds = PoseDataset(val_df, pose_root, clip_length=args.clip_length)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = PoseLSTMClassifier(
        input_dim=99,
        num_classes=len(class_names),
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        lstm_dropout=args.lstm_dropout,
        lstm_bidirectional=args.lstm_bidirectional,
        classifier_dropout=args.classifier_dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    criterion = nn.CrossEntropyLoss()
    report_dir = resolve_project_path(args.report_dir)
    evaluate_and_save(model, val_loader, criterion, device, class_names, report_dir, "val")


def overlay_video(args: argparse.Namespace) -> None:
    """Run inference on a video and save a visual overlay with MediaPipe landmarks."""
    device = resolve_device(args.device)
    checkpoint_path = resolve_project_path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, device)
    class_names = parse_class_names(args.class_names)
    
    # 1. Initialize Model
    model = PoseLSTMClassifier(
        input_dim=99,
        num_classes=len(class_names),
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        lstm_dropout=args.lstm_dropout,
        lstm_bidirectional=args.lstm_bidirectional,
        classifier_dropout=args.classifier_dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 2. Setup MediaPipe Landmarker
    model_task_path = resolve_project_path(args.pose_landmarker_path)
    if not model_task_path.exists():
        print(f"Error: MediaPipe task file not found at {model_task_path}")
        return

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_task_path)),
        running_mode=vision.RunningMode.IMAGE
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    # 3. Setup Video I/O
    input_path = resolve_project_path(args.video)
    if not input_path.exists():
        print(f"Error: Input video not found at {input_path}")
        return

    cap = cv2.VideoCapture(str(input_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Save output relative to the checkpoint's run directory
    run_dir = checkpoint_path.parent.parent
    output_dir = run_dir / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # 4. Processing Loop
    pose_history = deque(maxlen=args.clip_length)
    print(f"Processing video: {input_path.name}")
    print(f"Saving to: {output_path}")
    
    progress = tqdm(total=total_frames)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract Pose
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = landmarker.detect(mp_image)
        
        current_pose = [[0.0, 0.0, 0.0]] * 33
        if detection_result.pose_landmarks:
            current_pose = [[lm.x, lm.y, lm.z] for lm in detection_result.pose_landmarks[0]]
            # Draw skeleton
            for start_idx, end_idx in POSE_CONNECTIONS:
                start_lm = detection_result.pose_landmarks[0][start_idx]
                end_lm = detection_result.pose_landmarks[0][end_idx]
                p1 = (int(start_lm.x * width), int(start_lm.y * height))
                p2 = (int(end_lm.x * width), int(end_lm.y * height))
                cv2.line(frame, p1, p2, (0, 255, 0), 2)
            for lm in detection_result.pose_landmarks[0]:
                cv2.circle(frame, (int(lm.x * width), int(lm.y * height)), 3, (0, 0, 255), -1)

        pose_history.append(current_pose)
        
        # Predict
        label_text = "COLLECTING DATA..."
        color = (255, 255, 255)
        
        if len(pose_history) == args.clip_length:
            input_tensor = torch.tensor(list(pose_history)).float().to(device)
            input_tensor = input_tensor.view(1, args.clip_length, -1)
            
            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1)[0]
                pred_idx = torch.argmax(probs).item()
                confidence = probs[pred_idx].item()
                pred_class = class_names[pred_idx]
                
                if pred_class != "none" and confidence > args.confidence_threshold:
                    label_text = f"{pred_class.upper()} ({confidence:.2f})"
                    color = (0, 255, 0) # Green for action
                else:
                    label_text = "NONE"
                    color = (200, 200, 200) # Grey for none

        # Draw UI
        cv2.putText(frame, label_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        out.write(frame)
        progress.update(1)

    cap.release()
    out.release()
    landmarker.close()
    progress.close()
    print(f"\nOverlay complete! Video saved to {output_path}")

