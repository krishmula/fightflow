import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
import yaml
import json
import random
import os

# Constants
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# MediaPipe Pose Connections (33 landmarks)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
]

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def resolve_device(device_name: str = "auto") -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)

def resolve_project_path(path: str | Path) -> Path:
    return Path(path).resolve()

def get_data_config(config_path: Path | None = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "models" / "base" / "data_config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def parse_class_names(class_names: str) -> list[str]:
    return [c.strip() for c in class_names.split(",")]

def load_manifest(manifest_path: Path) -> pd.DataFrame:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    return pd.read_csv(manifest_path)

def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def compute_class_weights(labels: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor | None:
    if len(labels) == 0:
        return None
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = np.zeros_like(counts)
    non_zero = counts > 0
    weights[non_zero] = counts.sum() / (num_classes * counts[non_zero])
    return torch.tensor(weights, dtype=torch.float32, device=device)

class PoseDataset(Dataset):
    """Dataset that loads pre-extracted pose keypoints."""
    def __init__(self, df: pd.DataFrame, pose_root: Path, clip_length: int, transform=None, noise_std: float = 0.0):
        self.df = df.reset_index(drop=True)
        self.pose_root = Path(pose_root)
        self.clip_length = clip_length
        self.transform = transform
        self.noise_std = noise_std

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        label = int(row["label"])
        
        # New paradigm: use pose_path directly from manifest
        pose_path = Path(row["pose_path"])
        
        if not pose_path.exists():
            # Try resolving relative to project root if absolute fails
            pose_path = resolve_project_path(row["pose_path"])
            if not pose_path.exists():
                raise FileNotFoundError(f"Pose data not found at: {pose_path}")
            
        # Load pose sequence: (N_frames, 33, 3)
        poses = np.load(pose_path)
        
        # Ensure correct clip length
        total_frames = poses.shape[0]
        if total_frames != self.clip_length:
            if total_frames > self.clip_length:
                indices = np.linspace(0, total_frames - 1, self.clip_length).astype(int)
                poses = poses[indices]
            else:
                padding = np.tile(poses[-1:], (self.clip_length - total_frames, 1, 1))
                poses = np.concatenate([poses, padding], axis=0)
        
        # Convert to tensor
        poses_tensor = torch.from_numpy(poses).float()
        
        # Apply noise augmentation if training
        if self.noise_std > 0:
            noise = torch.randn_like(poses_tensor) * self.noise_std
            poses_tensor = poses_tensor + noise
            
        # Flatten joints (33*3 = 99)
        if len(poses_tensor.shape) == 3:
            poses_tensor = poses_tensor.view(self.clip_length, -1)
            
        return poses_tensor, label

@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    macro_f1: float

def run_epoch(loader, model, criterion, device, optimizer=None, max_grad_norm=1.0):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    losses, y_t, y_p = [], [], []
    
    progress = tqdm(loader, leave=False, unit="batch")
    for poses, targets in progress:
        poses, targets = poses.to(device), targets.to(device)
        
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            
        with torch.set_grad_enabled(is_train):
            logits = model(poses)
            loss = criterion(logits, targets)
            
        if is_train:
            loss.backward()
            if max_grad_norm:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            
        losses.append(float(loss.item()))
        y_t.extend(targets.cpu().tolist())
        y_p.extend(torch.argmax(logits, 1).cpu().tolist())
        progress.set_postfix(loss=np.mean(losses[-10:]))
        
    from sklearn.metrics import accuracy_score, f1_score
    return EpochMetrics(
        loss=float(np.mean(losses)),
        accuracy=accuracy_score(y_t, y_p),
        macro_f1=f1_score(y_t, y_p, average="macro", zero_division=0)
    ), np.array(y_t), np.array(y_p)

def save_checkpoint(**kwargs):
    path = kwargs.pop("path")
    model = kwargs.pop("model")
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {"model_state_dict": model.state_dict()}
    ckpt.update(kwargs)
    torch.save(ckpt, path)

def load_checkpoint(path, device):
    return torch.load(path, map_location=device, weights_only=False)

def split_by_video(df, splits, seed=42):
    if "split" in df.columns:
        unique_splits = df["split"].dropna().unique()
        if all(s in unique_splits for s in ["train", "val", "test"]):
            return df[df["split"] == "train"], df[df["split"] == "val"], df[df["split"] == "test"]
            
    v_ids = sorted(df["video_id"].unique())
    random.Random(seed).shuffle(v_ids)
    n = len(v_ids)
    t_e = int(n * splits.get("train", 0.7))
    v_e = t_e + int(n * splits.get("val", 0.15))
    
    t_ids = set(v_ids[:t_e])
    v_ids_s = set(v_ids[t_e:v_e])
    te_ids = set(v_ids[v_e:])
    
    df_t = df[df["video_id"].isin(t_ids)].copy()
    df_v = df[df["video_id"].isin(v_ids_s)].copy()
    df_te = df[df["video_id"].isin(te_ids)].copy()
    
    for d, s in zip([df_t, df_v, df_te], ["train", "val", "test"]):
        d["split"] = s
    return df_t, df_v, df_te

def evaluate_and_save(model, loader, criterion, device, class_names, out_dir, split_name):
    metrics, y_true, y_pred = run_epoch(loader, model, criterion, device)
    
    from sklearn.metrics import classification_report, confusion_matrix
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = {
        "metrics": {"loss": metrics.loss, "accuracy": metrics.accuracy, "macro_f1": metrics.macro_f1},
        "classification_report": classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    }
    save_json(out_dir / f"{split_name}_metrics.json", rep)
    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(out_dir / f"{split_name}_confusion_matrix.csv")
    return rep["metrics"]
