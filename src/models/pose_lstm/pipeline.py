#!/usr/bin/env python3
"""Train and evaluate a Pose-based LSTM classifier."""

from __future__ import annotations
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


# ── Model ──────────────────────────────────────────────────────────────────

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.attn(x), dim=1)  # (B, T, 1)
        return torch.sum(x * weights, dim=1)


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
        self.attention  = TemporalAttention(lstm_out_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(p=classifier_dropout),
            nn.Linear(lstm_out_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        return self.classifier(self.attention(lstm_out))


# ── Shared helpers ─────────────────────────────────────────────────────────

def _build_model(args, num_classes: int, device: torch.device) -> PoseLSTMClassifier:
    """Construct PoseLSTMClassifier from hparam args. Single source of truth."""
    return PoseLSTMClassifier(
        input_dim=args.input_dim,
        num_classes=num_classes,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        lstm_dropout=args.lstm_dropout,
        lstm_bidirectional=args.lstm_bidirectional,
        classifier_dropout=args.classifier_dropout,
    ).to(device)


def _make_loader(df, args, noise_std: float = 0.0) -> DataLoader:
    """Create a DataLoader for a pose split, forwarding landmark config from args."""
    ds = PoseDataset(
        df,
        pose_root=resolve_project_path(args.pose_root),
        clip_length=args.clip_length,
        noise_std=noise_std,
        landmark_cfg=getattr(args, "landmark_indices", None),
        extension_frames=getattr(args, "extension_frames", 4),
    )
    return DataLoader(ds, batch_size=args.batch_size, shuffle=(noise_std > 0),
                      num_workers=args.num_workers)


def _resolve_manifest(args) -> Path:
    """Return the resolved pose manifest path, preferring CLI arg over data config."""
    config = get_data_config()
    data_cfg = config.get("data", {})
    return resolve_project_path(
        getattr(args, "manifest_file", "") or
        data_cfg.get("pose_manifest_file", "data/processed/pose_manifest.csv")
    )


def _inference_features(poses_array: np.ndarray, lm: dict, clip_length: int,
                         extension_frames: int) -> np.ndarray:
    """Build the same 222-dim feature vector used by PoseDataset (numpy version).

    Used inside overlay_video so the inference path stays in sync with training.
    """
    # ── Normalize absolute coordinates ──────────────────────────────
    # Center all coordinates relative to the mid-hip to remove screen position variance
    l_hip, r_hip = lm["l_hip"], lm["r_hip"]
    mid_hip = (poses_array[:, l_hip, :] + poses_array[:, r_hip, :]) / 2.0  # (seq, 3)
    poses_array = poses_array - mid_hip[:, np.newaxis, :]  # (seq, 33, 3)

    poses_flat = poses_array.reshape(clip_length, -1)  # (seq, 99)

    velocities = np.zeros_like(poses_flat)
    velocities[1:] = poses_flat[1:] - poses_flat[:-1]

    def rel_np(p1: int, p2: int) -> np.ndarray:
        return poses_array[:, p1, :] - poses_array[:, p2, :]

    rel_vectors = np.concatenate([
        rel_np(lm["l_wrist"], lm["l_shoulder"]),
        rel_np(lm["r_wrist"], lm["r_shoulder"]),
        rel_np(lm["l_wrist"], lm["l_hip"]),
        rel_np(lm["r_wrist"], lm["r_hip"]),
        rel_np(lm["l_wrist"], lm["nose"]),
        rel_np(lm["r_wrist"], lm["nose"]),
    ], axis=-1)  # (seq, 18)

    ext_start = max(0, clip_length - extension_frames)
    l_wr, r_wr = lm["l_wrist"], lm["r_wrist"]
    l_ext = np.mean(velocities[ext_start:, l_wr * 3: l_wr * 3 + 3], axis=0, keepdims=True)
    r_ext = np.mean(velocities[ext_start:, r_wr * 3: r_wr * 3 + 3], axis=0, keepdims=True)
    l_ext_feat = np.tile(l_ext, (clip_length, 1))
    r_ext_feat = np.tile(r_ext, (clip_length, 1))

    # Max Absolute Velocity over the entire clip
    l_max_v = np.max(np.abs(velocities[:, l_wr * 3: l_wr * 3 + 3]), axis=0, keepdims=True)
    r_max_v = np.max(np.abs(velocities[:, r_wr * 3: r_wr * 3 + 3]), axis=0, keepdims=True)
    l_max_v_feat = np.tile(l_max_v, (clip_length, 1))
    r_max_v_feat = np.tile(r_max_v, (clip_length, 1))

    # ── Joint Angles (Elbow Bend) ───────────────────────────────────
    l_sh, r_sh = lm["l_shoulder"], lm["r_shoulder"]
    l_el, r_el = lm.get("l_elbow", 13), lm.get("r_elbow", 14)

    def get_angle_np(sh, el, wr):
        v1 = poses_array[:, sh, :] - poses_array[:, el, :]
        v2 = poses_array[:, wr, :] - poses_array[:, el, :]
        dot = np.sum(v1 * v2, axis=1)
        norm_v1 = np.linalg.norm(v1, axis=1)
        norm_v2 = np.linalg.norm(v2, axis=1)
        cos_theta = dot / (norm_v1 * norm_v2 + 1e-6)
        return np.arccos(np.clip(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6))[:, np.newaxis]

    l_angle_feat = get_angle_np(l_sh, l_el, l_wr)
    r_angle_feat = get_angle_np(r_sh, r_el, r_wr)

    # ── Wrist-to-Face Velocity ──────────────────────────────────────
    nose = lm["nose"]
    l_wr_nose = poses_array[:, l_wr, :] - poses_array[:, nose, :]
    r_wr_nose = poses_array[:, r_wr, :] - poses_array[:, nose, :]
    
    l_wr_nose_vel = np.zeros_like(l_wr_nose)
    r_wr_nose_vel = np.zeros_like(r_wr_nose)
    l_wr_nose_vel[1:] = l_wr_nose[1:] - l_wr_nose[:-1]
    r_wr_nose_vel[1:] = r_wr_nose[1:] - r_wr_nose[:-1]

    return np.concatenate([poses_flat, velocities, rel_vectors,
                            l_ext_feat, r_ext_feat,
                            l_max_v_feat, r_max_v_feat,
                            l_angle_feat, r_angle_feat,
                            l_wr_nose_vel, r_wr_nose_vel], axis=-1)  # (seq, 236)


# ── Tasks ──────────────────────────────────────────────────────────────────

def train(args) -> dict:
    class_names = parse_class_names(args.class_names)
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    manifest_path = _resolve_manifest(args)
    df = load_manifest(manifest_path)
    train_df, val_df, test_df = split_by_video(df, args.splits, args.seed)

    train_loader = _make_loader(train_df, args, noise_std=getattr(args, "pose_noise_std", 0.0))
    val_loader   = _make_loader(val_df,   args)
    test_loader  = _make_loader(test_df,  args)

    model = _build_model(args, len(class_names), device)

    resume_path = getattr(args, "resume", None)
    if resume_path:
        print(f"Resuming from: {resume_path}")
        ckpt = load_checkpoint(resolve_project_path(resume_path), device)
        model.load_state_dict(ckpt["model_state_dict"])

    label_smoothing = getattr(args, "label_smoothing", 0.0)
    class_weights   = compute_class_weights(
        train_df["label"].values, len(class_names), device
    ) if args.use_class_weights else None
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=args.lr_patience
    )

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir     = resolve_project_path(args.output_dir) / f"{args.run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    report_dir     = run_dir / "reports"

    best_macro_f1, best_epoch, no_improve = -1.0, -1, 0
    history_rows: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_m, _, _ = run_epoch(train_loader, model, criterion, device, optimizer)
        val_m,   _, _ = run_epoch(val_loader,   model, criterion, device)
        scheduler.step(val_m.macro_f1)

        print(
            "train_loss={:.4f} train_acc={:.4f} train_f1={:.4f} | "
            "val_loss={:.4f} val_acc={:.4f} val_f1={:.4f}".format(
                train_m.loss, train_m.accuracy, train_m.macro_f1,
                val_m.loss,   val_m.accuracy,   val_m.macro_f1,
            )
        )

        save_checkpoint(checkpoint_dir / "last.pt", model, epoch=epoch,
                        best_val_macro_f1=max(best_macro_f1, val_m.macro_f1))

        if val_m.macro_f1 > best_macro_f1:
            best_macro_f1, best_epoch, no_improve = val_m.macro_f1, epoch, 0
            save_checkpoint(checkpoint_dir / "best.pt", model, epoch=epoch,
                            best_val_macro_f1=best_macro_f1)
        else:
            no_improve += 1

        history_rows.append({
            "epoch": epoch, "train_loss": train_m.loss,
            "train_acc": train_m.accuracy, "train_f1": train_m.macro_f1,
            "val_loss": val_m.loss, "val_acc": val_m.accuracy, "val_f1": val_m.macro_f1,
        })

        if no_improve >= args.patience:
            print(f"Early stopping after {args.patience} epochs without improvement.")
            break

    history_path = run_dir / "history.csv"
    if history_rows:
        with history_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history_rows[0].keys()))
            writer.writeheader()
            writer.writerows(history_rows)

    print(f"\nBest val Macro-F1: {best_macro_f1:.4f} at epoch {best_epoch}")

    best_ckpt = load_checkpoint(checkpoint_dir / "best.pt", device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    if len(val_loader.dataset) > 0:
        val_summary = evaluate_and_save(model, val_loader, criterion, device,
                                         class_names, report_dir, "val")
        print("\nValidation summary:", json.dumps(val_summary, indent=2))

    if len(test_loader.dataset) > 0:
        test_summary = evaluate_and_save(model, test_loader, criterion, device,
                                          class_names, report_dir, "test")
        print("Test summary:", json.dumps(test_summary, indent=2))
    else:
        print("\nSkipping test evaluation: test set is empty.")

    summary = {
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro_f1,
        "checkpoint_best": str(checkpoint_dir / "best.pt"),
        "report_dir": str(report_dir),
    }
    save_json(run_dir / "run_summary.json", summary)
    print(f"\nRun complete. Artifacts saved to: {run_dir}")
    return summary


def _load_eval_checkpoint(args, device):
    """Shared setup for test_only / validate_only."""
    checkpoint_path = resolve_project_path(args.checkpoint)
    checkpoint      = load_checkpoint(checkpoint_path, device)
    class_names     = parse_class_names(args.class_names)
    manifest_path   = _resolve_manifest(args)
    df              = load_manifest(manifest_path)
    return checkpoint, class_names, df, checkpoint_path


def test_only(args) -> None:
    device = resolve_device(args.device)
    checkpoint, class_names, df, _ = _load_eval_checkpoint(args, device)
    _, _, test_df = split_by_video(df, args.splits, args.seed)

    model = _build_model(args, len(class_names), device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loader  = _make_loader(test_df, args)
    criterion    = nn.CrossEntropyLoss()
    report_dir   = resolve_project_path(args.report_dir)
    evaluate_and_save(model, test_loader, criterion, device, class_names, report_dir, "test")


def validate_only(args) -> None:
    device = resolve_device(args.device)
    checkpoint, class_names, df, _ = _load_eval_checkpoint(args, device)
    _, val_df, _ = split_by_video(df, args.splits, args.seed)

    model = _build_model(args, len(class_names), device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_loader  = _make_loader(val_df, args)
    criterion   = nn.CrossEntropyLoss()
    report_dir  = resolve_project_path(args.report_dir)
    evaluate_and_save(model, val_loader, criterion, device, class_names, report_dir, "val")


def overlay_video(args) -> None:
    """Run inference on a video and save a skeleton-overlay output video."""
    device          = resolve_device(args.device)
    checkpoint_path = resolve_project_path(args.checkpoint)
    checkpoint      = load_checkpoint(checkpoint_path, device)
    class_names     = parse_class_names(args.class_names)
    lm              = getattr(args, "landmark_indices", None) or {
        "nose": 0, "l_shoulder": 11, "r_shoulder": 12,
        "l_wrist": 15, "r_wrist": 16, "l_hip": 23, "r_hip": 24,
    }
    extension_frames = getattr(args, "extension_frames", 4)

    model = _build_model(args, len(class_names), device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    model_task_path = resolve_project_path(args.pose_landmarker_path)
    if not model_task_path.exists():
        print(f"Error: MediaPipe task file not found at {model_task_path}")
        return

    options    = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_task_path)),
        running_mode=vision.RunningMode.IMAGE,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    input_path = resolve_project_path(args.video)
    if not input_path.exists():
        print(f"Error: Input video not found at {input_path}")
        return

    cap         = cv2.VideoCapture(str(input_path))
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps         = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    run_dir    = checkpoint_path.parent.parent
    output_dir = run_dir / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    pose_history: deque = deque(maxlen=args.clip_length)
    print(f"Processing video: {input_path.name}  →  {output_path}")

    for _ in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image       = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result         = landmarker.detect(mp_image)
        current_pose   = [[0.0, 0.0, 0.0]] * 33

        if result.pose_landmarks:
            current_pose = [[lmk.x, lmk.y, lmk.z] for lmk in result.pose_landmarks[0]]
            for s, e in POSE_CONNECTIONS:
                p1 = (int(result.pose_landmarks[0][s].x * width),
                      int(result.pose_landmarks[0][s].y * height))
                p2 = (int(result.pose_landmarks[0][e].x * width),
                      int(result.pose_landmarks[0][e].y * height))
                cv2.line(frame, p1, p2, (0, 255, 0), 2)
            for lmk in result.pose_landmarks[0]:
                cv2.circle(frame, (int(lmk.x * width), int(lmk.y * height)), 3, (0, 0, 255), -1)

        pose_history.append(current_pose)

        label_text, color = "COLLECTING DATA...", (255, 255, 255)

        if len(pose_history) == args.clip_length:
            poses_array  = np.array(list(pose_history))  # (seq, 33, 3)
            combined     = _inference_features(poses_array, lm, args.clip_length, extension_frames)
            input_tensor = torch.from_numpy(combined).float().unsqueeze(0).to(device)

            with torch.no_grad():
                logits     = model(input_tensor)
                probs      = torch.softmax(logits, dim=1)[0]
                pred_idx   = torch.argmax(probs).item()
                confidence = probs[pred_idx].item()
                pred_class = class_names[pred_idx]

            if pred_class != "none" and confidence > args.confidence_threshold:
                label_text = f"{pred_class.upper()} ({confidence:.2f})"
                color      = (0, 255, 0)
            else:
                label_text, color = "NONE", (200, 200, 200)

        cv2.putText(frame, label_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        out.write(frame)

    cap.release()
    out.release()
    landmarker.close()
    print(f"\nOverlay complete! Saved to {output_path}")
