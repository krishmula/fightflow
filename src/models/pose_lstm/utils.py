"""pose_lstm-specific utilities.

Shared helpers (set_seed, resolve_device, save_json, EpochMetrics, etc.)
live in src/models/base/utils.py and are re-exported from here so that
existing pipeline.py imports remain unchanged.
"""
from __future__ import annotations

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm

# ── Re-export everything shared so pipeline.py imports stay identical ──────
from ..base.utils import (
    EpochMetrics,
    compute_class_weights,
    get_data_config,
    load_checkpoint,
    load_manifest,
    load_yaml,
    parse_class_names,
    resolve_device,
    resolve_project_path,
    save_checkpoint,
    save_json,
    set_seed,
    split_by_video,
)

# Re-export run_epoch under the same name after defining the pose version
# (done at the bottom of this file — keep the import chain intact)

# ── Pose-LSTM-specific constants ───────────────────────────────────────────

# MediaPipe Pose skeleton connections (33-landmark model)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]

# Default MediaPipe landmark indices (matches hparams.yaml landmark_indices)
_DEFAULT_LANDMARK_CFG: dict[str, int] = {
    "nose": 0,
    "l_shoulder": 11,
    "r_shoulder": 12,
    "l_wrist": 15,
    "r_wrist": 16,
    "l_hip": 23,
    "r_hip": 24,
}


# ── PoseDataset ────────────────────────────────────────────────────────────

class PoseDataset(Dataset):
    """Dataset that loads pre-extracted MediaPipe pose keypoints (.npy files).

    Feature vector per frame (all dims configurable via landmark_cfg /
    extension_frames so no magic numbers live in this class):

        99  raw coordinates  (33 landmarks × 3)
      + 99  frame velocities (finite differences)
      + 18  wrist→keypoint relative vectors (6 pairs × 3)
      +  3  L-wrist extension-phase avg velocity
      +  3  R-wrist extension-phase avg velocity
      ────
       222  total  (stored in hparams.yaml as input_dim)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        pose_root: Path,
        clip_length: int,
        transform=None,
        noise_std: float = 0.0,
        landmark_cfg: dict[str, int] | None = None,
        extension_frames: int = 4,
    ):
        self.df = df.reset_index(drop=True)
        self.pose_root = Path(pose_root)
        self.clip_length = clip_length
        self.transform = transform
        self.noise_std = noise_std
        self.lm = landmark_cfg if landmark_cfg is not None else _DEFAULT_LANDMARK_CFG
        self.extension_frames = extension_frames

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        label = int(row["label"])

        pose_path = Path(row["pose_path"])
        if not pose_path.exists():
            pose_path = resolve_project_path(row["pose_path"])
            if not pose_path.exists():
                raise FileNotFoundError(f"Pose data not found: {pose_path}")

        # Load (N_frames, 33, 3)
        poses = np.load(pose_path)

        # ── Clip normalisation ──────────────────────────────────────────
        total = poses.shape[0]
        if total != self.clip_length:
            if total > self.clip_length:
                idx = np.linspace(0, total - 1, self.clip_length).astype(int)
                poses = poses[idx]
            else:
                pad = np.tile(poses[-1:], (self.clip_length - total, 1, 1))
                poses = np.concatenate([poses, pad], axis=0)

        poses_t = torch.from_numpy(poses).float()  # (seq, 33, 3)

        # ── Augmentation ────────────────────────────────────────────────
        if self.noise_std > 0:
            # Temporal Shift Augmentation
            # Shift the 11-frame window forward or backward by 1 or 2 frames to prevent exact timing memorization
            shift = random.choice([-2, -1, 0, 1, 2])
            if shift > 0:
                # Shift forward: pad end
                pad = poses_t[-1:].repeat(shift, 1, 1)
                poses_t = torch.cat([poses_t[shift:], pad], dim=0)
            elif shift < 0:
                # Shift backward: pad beginning
                shift = abs(shift)
                pad = poses_t[0:1].repeat(shift, 1, 1)
                poses_t = torch.cat([pad, poses_t[:-shift]], dim=0)

            # Spatial Augmentation
            poses_t = poses_t + torch.randn_like(poses_t) * self.noise_std
            scale_x = random.uniform(0.9, 1.1)
            poses_t[:, :, 0] *= scale_x

        # ── Feature engineering ─────────────────────────────────────────
        lm = self.lm
        nose, l_sh, r_sh = lm["nose"], lm["l_shoulder"], lm["r_shoulder"]
        l_el, r_el       = lm.get("l_elbow", 13), lm.get("r_elbow", 14)
        l_wr, r_wr       = lm["l_wrist"],  lm["r_wrist"]
        l_hip, r_hip     = lm["l_hip"],    lm["r_hip"]

        # ── Normalize absolute coordinates ──────────────────────────────
        # Center all coordinates relative to the mid-hip to remove screen position variance
        mid_hip = (poses_t[:, l_hip, :] + poses_t[:, r_hip, :]) / 2.0  # (seq, 3)
        poses_t = poses_t - mid_hip.unsqueeze(1)  # (seq, 33, 3)

        def rel(p1: int, p2: int) -> torch.Tensor:
            return poses_t[:, p1, :] - poses_t[:, p2, :]  # (seq, 3)

        rel_vectors = torch.cat([
            rel(l_wr, l_sh), rel(r_wr, r_sh),
            rel(l_wr, l_hip), rel(r_wr, r_hip),
            rel(l_wr, nose),  rel(r_wr, nose),
        ], dim=-1)  # (seq, 18)

        poses_flat = poses_t.view(self.clip_length, -1)  # (seq, 99)

        velocities = torch.zeros_like(poses_flat)
        velocities[1:] = poses_flat[1:] - poses_flat[:-1]  # (seq, 99)

        # Extension-phase avg wrist velocity (last `extension_frames` frames)
        ext_start = max(0, self.clip_length - self.extension_frames)
        l_ext = velocities[ext_start:, l_wr * 3: l_wr * 3 + 3].mean(dim=0)
        r_ext = velocities[ext_start:, r_wr * 3: r_wr * 3 + 3].mean(dim=0)
        l_ext_feat = l_ext.unsqueeze(0).expand(self.clip_length, -1)  # (seq, 3)
        r_ext_feat = r_ext.unsqueeze(0).expand(self.clip_length, -1)  # (seq, 3)

        # Max Absolute Velocity over the entire clip (captures X-heavy hooks vs Z-heavy straights)
        l_max_v = torch.max(torch.abs(velocities[:, l_wr * 3: l_wr * 3 + 3]), dim=0)[0]  # (3,)
        r_max_v = torch.max(torch.abs(velocities[:, r_wr * 3: r_wr * 3 + 3]), dim=0)[0]  # (3,)
        l_max_v_feat = l_max_v.unsqueeze(0).expand(self.clip_length, -1)  # (seq, 3)
        r_max_v_feat = r_max_v.unsqueeze(0).expand(self.clip_length, -1)  # (seq, 3)

        # ── Joint Angles (Elbow Bend) ───────────────────────────────────
        def get_angle(sh, el, wr):
            v1 = poses_t[:, sh, :] - poses_t[:, el, :]
            v2 = poses_t[:, wr, :] - poses_t[:, el, :]
            cos_theta = torch.sum(v1 * v2, dim=1) / (torch.norm(v1, dim=1) * torch.norm(v2, dim=1) + 1e-6)
            # Clamp for numerical stability before acos
            return torch.acos(torch.clamp(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)).unsqueeze(1) # (seq, 1)

        l_angle_feat = get_angle(l_sh, l_el, l_wr)
        r_angle_feat = get_angle(r_sh, r_el, r_wr)

        # ── Wrist-to-Face Velocity ──────────────────────────────────────
        # Isolates arm punch velocity from whole-body movement (e.g. jumping back in a guard)
        l_wr_nose = poses_t[:, l_wr, :] - poses_t[:, nose, :]
        r_wr_nose = poses_t[:, r_wr, :] - poses_t[:, nose, :]
        l_wr_nose_vel = torch.zeros_like(l_wr_nose)
        r_wr_nose_vel = torch.zeros_like(r_wr_nose)
        l_wr_nose_vel[1:] = l_wr_nose[1:] - l_wr_nose[:-1]
        r_wr_nose_vel[1:] = r_wr_nose[1:] - r_wr_nose[:-1]

        # Final: (seq, 99 + 99 + 18 + 3 + 3 + 3 + 3 + 1 + 1 + 3 + 3) = (seq, 236)
        return torch.cat([poses_flat, velocities, rel_vectors,
                          l_ext_feat, r_ext_feat,
                          l_max_v_feat, r_max_v_feat,
                          l_angle_feat, r_angle_feat,
                          l_wr_nose_vel, r_wr_nose_vel], dim=-1), label


# ── Epoch loop (no AMP — pose features are lightweight CPU tensors) ────────

def run_epoch(loader, model, criterion, device, optimizer=None, max_grad_norm=1.0):
    """Single train or eval pass over *loader*. Returns (EpochMetrics, y_true, y_pred)."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    losses, y_t, y_p = [], [], []

    for poses, targets in tqdm(loader, leave=False, unit="batch"):
        poses, targets = poses.to(device), targets.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            logits = model(poses)
            loss   = criterion(logits, targets)
        if is_train:
            loss.backward()
            if max_grad_norm:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        losses.append(float(loss.item()))
        y_t.extend(targets.cpu().tolist())
        y_p.extend(torch.argmax(logits, 1).cpu().tolist())

    from sklearn.metrics import accuracy_score, f1_score
    avg_loss = float(np.mean(losses)) if losses else 0.0
    if not y_t:
        return EpochMetrics(loss=avg_loss, accuracy=0.0, macro_f1=0.0), np.array([]), np.array([])
    return EpochMetrics(
        loss=avg_loss,
        accuracy=accuracy_score(y_t, y_p),
        macro_f1=f1_score(y_t, y_p, average="macro", zero_division=0),
    ), np.array(y_t), np.array(y_p)


# ── Evaluation helper ──────────────────────────────────────────────────────

def evaluate_and_save(model, loader, criterion, device, class_names, out_dir, split_name):
    """Run a full eval pass and write metrics + confusion matrix to *out_dir*."""
    metrics, y_true, y_pred = run_epoch(loader, model, criterion, device)

    if len(y_true) == 0:
        print(f"Warning: no samples in '{split_name}' loader — skipping reports.")
        return {"loss": 0.0, "accuracy": 0.0, "macro_f1": 0.0}

    from sklearn.metrics import classification_report, confusion_matrix
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rep = {
        "metrics": {
            "loss": metrics.loss,
            "accuracy": metrics.accuracy,
            "macro_f1": metrics.macro_f1,
        },
        "classification_report": classification_report(
            y_true, y_pred, target_names=class_names,
            output_dict=True, zero_division=0,
        ),
    }
    save_json(out_dir / f"{split_name}_metrics.json", rep)
    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        out_dir / f"{split_name}_confusion_matrix.csv"
    )
    return rep["metrics"]
