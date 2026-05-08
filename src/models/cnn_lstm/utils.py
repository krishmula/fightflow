#!/usr/bin/env python3
"""Shared utilities for CNN+LSTM training and evaluation."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (project_root() / path).resolve()


def resolve_latest_checkpoint(backbone: str) -> Path | None:
    runs_dir = resolve_project_path(f"runs/{backbone}")
    if not runs_dir.is_dir():
        return None

    candidates = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for run_dir in candidates:
        best_path = run_dir / "checkpoints" / "best.pt"
        if best_path.is_file():
            return best_path
    return None


def get_data_config() -> dict:
    config_path = Path(__file__).parents[1] / "base" / "data_config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def parse_class_names(raw: str) -> list[str]:
    names = [s.strip().lower() for s in raw.split(",") if s.strip()]
    if not names:
        raise ValueError("Class list cannot be empty")
    return names


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def get_transforms(image_size: int, augment: bool) -> transforms.Compose:
    config = get_data_config()
    aug_cfg = config.get("augmentation", {})

    norm_cfg = aug_cfg.get("normalize", {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]})
    normalize = transforms.Normalize(mean=norm_cfg["mean"], std=norm_cfg["std"])

    tf = [transforms.Resize((image_size, image_size))]

    if augment:
        flip_p = aug_cfg.get("random_horizontal_flip_p", 0.5)
        jitter_cfg = aug_cfg.get("color_jitter", {"brightness": 0.2, "contrast": 0.2, "saturation": 0.15})
        rotation_cfg = aug_cfg.get("random_rotation", {"degrees": 15})
        affine_cfg = aug_cfg.get(
            "random_affine",
            {"degrees": 0, "translate": [0.1, 0.1], "scale": [0.9, 1.1], "shear": 5},
        )
        erasing_cfg = aug_cfg.get("random_erasing", {"p": 0.0, "scale": [0.02, 0.1], "ratio": [0.3, 3.3]})

        tf.append(transforms.RandomHorizontalFlip(p=flip_p))
        tf.append(
            transforms.ColorJitter(
                brightness=jitter_cfg["brightness"],
                contrast=jitter_cfg["contrast"],
                saturation=jitter_cfg["saturation"],
            )
        )
        if rotation_cfg.get("degrees", 0) > 0:
            tf.append(transforms.RandomRotation(degrees=rotation_cfg["degrees"]))
        tf.append(
            transforms.RandomAffine(
                degrees=affine_cfg["degrees"],
                translate=affine_cfg["translate"],
                scale=affine_cfg["scale"],
                shear=affine_cfg["shear"],
            )
        )

        tf.extend([transforms.ToTensor(), normalize])
        if erasing_cfg.get("p", 0) > 0:
            tf.append(
                transforms.RandomErasing(
                    p=erasing_cfg["p"],
                    scale=erasing_cfg["scale"],
                    ratio=erasing_cfg["ratio"],
                )
            )
        return transforms.Compose(tf)

    tf.extend([transforms.ToTensor(), normalize])
    return transforms.Compose(tf)


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Clip manifest not found: {manifest_path}")
    df = pd.read_csv(manifest_path)
    required = {"clip_dir", "label", "video_id"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Manifest missing columns: {sorted(missing)}")
    df["clip_dir"] = df["clip_dir"].astype(str)
    df["video_id"] = df["video_id"].astype(str)
    df["label"] = df["label"].astype(int)
    return df


def split_by_video(df: pd.DataFrame, splits: dict, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_ratio = float(splits.get("train", 0.7))
    val_ratio = float(splits.get("val", 0.15))
    test_ratio = float(splits.get("test", 0.15))

    video_ids = sorted(df["video_id"].unique().tolist())
    if len(video_ids) < 2:
        return df, df.iloc[0:0].copy(), df.iloc[0:0].copy()

    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio <= 0:
        return df, df.iloc[0:0].copy(), df.iloc[0:0].copy()

    split_ratios = {
        "train": train_ratio / total_ratio,
        "val": val_ratio / total_ratio,
        "test": test_ratio / total_ratio,
    }
    split_ratios = {k: v for k, v in split_ratios.items() if v > 0}

    labels = sorted(df["label"].unique().tolist())
    video_label_counts = (
        df.groupby(["video_id", "label"]).size().unstack(fill_value=0).reindex(video_ids, fill_value=0)
    )
    video_totals = video_label_counts.sum(axis=1)

    total_samples = float(video_totals.sum())
    targets_total = {k: total_samples * v for k, v in split_ratios.items()}
    label_totals = df["label"].value_counts().to_dict()
    targets_labels = {
        label: {k: float(label_totals.get(label, 0)) * split_ratios[k] for k in split_ratios}
        for label in labels
    }

    rng = random.Random(seed)
    assignments = {k: [] for k in split_ratios}
    current_total = {k: 0.0 for k in split_ratios}
    current_labels = {label: {k: 0.0 for k in split_ratios} for label in labels}

    shuffled = video_ids[:]
    rng.shuffle(shuffled)
    shuffled.sort(key=lambda vid: (video_totals.loc[vid], str(vid)), reverse=True)

    def cost(split_name: str, vid_counts: dict, vid_total: float) -> float:
        target_total = targets_total.get(split_name, 0.0)
        if target_total <= 0:
            return float("inf")

        total_after = current_total[split_name] + vid_total
        total_penalty = ((total_after - target_total) / target_total) ** 2

        label_penalty = 0.0
        label_count = 0
        for label in labels:
            target_label = targets_labels[label].get(split_name, 0.0)
            if target_label <= 0:
                continue
            label_count += 1
            label_after = current_labels[label][split_name] + float(vid_counts.get(label, 0))
            label_penalty += ((label_after - target_label) / target_label) ** 2

        if label_count:
            label_penalty /= label_count
        return total_penalty + label_penalty

    def fill_ratio(split_name: str) -> float:
        target_total = targets_total.get(split_name, 0.0)
        if target_total <= 0:
            return float("inf")
        return current_total[split_name] / target_total

    for vid in shuffled:
        vid_counts = video_label_counts.loc[vid].to_dict()
        vid_total = float(video_totals.loc[vid])
        candidates = list(split_ratios.keys())
        best_split = min(
            candidates,
            key=lambda s: (cost(s, vid_counts, vid_total), fill_ratio(s), current_total[s]),
        )
        assignments[best_split].append(vid)
        current_total[best_split] += vid_total
        for label in labels:
            current_labels[label][best_split] += float(vid_counts.get(label, 0))

    train_vids = assignments.get("train", [])
    val_vids = assignments.get("val", [])
    test_vids = assignments.get("test", [])

    train_df = df[df["video_id"].isin(train_vids)].reset_index(drop=True)
    val_df = df[df["video_id"].isin(val_vids)].reset_index(drop=True)
    test_df = df[df["video_id"].isin(test_vids)].reset_index(drop=True)
    return train_df, val_df, test_df


def compute_class_weights(labels: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor | None:
    if len(labels) == 0:
        return None
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = np.zeros_like(counts)
    non_zero = counts > 0
    weights[non_zero] = counts.sum() / (num_classes * counts[non_zero])
    return torch.tensor(weights, dtype=torch.float32, device=device)


def select_frame_indices(total: int, clip_length: int) -> list[int]:
    if total <= 0:
        return [0] * clip_length
    if total == clip_length:
        return list(range(total))
    if total > clip_length:
        indices = np.linspace(0, total - 1, clip_length)
        return [int(round(i)) for i in indices]
    return list(range(total)) + [total - 1] * (clip_length - total)


class ClipDataset(Dataset):
    def __init__(self, df: pd.DataFrame, clip_length: int, transform=None):
        self.df = df.reset_index(drop=True)
        self.clip_length = clip_length
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        clip_dir = Path(row["clip_dir"])
        label = int(row["label"])

        frame_paths = sorted(
            [p for p in clip_dir.iterdir() if p.suffix.lower() in SUPPORTED_IMAGE_EXTS]
        )
        if not frame_paths:
            raise RuntimeError(f"No frames found in clip dir: {clip_dir}")

        indices = select_frame_indices(len(frame_paths), self.clip_length)
        frames = []
        for idx in indices:
            frame_path = frame_paths[idx]
            try:
                image = Image.open(frame_path).convert("RGB")
            except (UnidentifiedImageError, OSError) as exc:
                raise RuntimeError(f"Could not read image: {frame_path}") from exc
            if self.transform is not None:
                image = self.transform(image)
            frames.append(image)

        clip = torch.stack(frames, dim=0)
        return clip, label


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    macro_f1: float


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[EpochMetrics, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if is_train else None

    progress = tqdm(loader, leave=False, unit="batch")
    for clips, targets in progress:
        clips = clips.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(clips)
                loss = criterion(logits, targets)

            if is_train and scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        losses.append(float(loss.detach().item()))
        preds = torch.argmax(logits, dim=1)

        y_true.extend(targets.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())

        progress.set_postfix(loss=f"{loss.item():.4f}")

    y_true_np = np.array(y_true, dtype=np.int64)
    y_pred_np = np.array(y_pred, dtype=np.int64)

    mean_loss = float(np.mean(losses)) if losses else 0.0
    acc = float(accuracy_score(y_true_np, y_pred_np)) if len(y_true_np) else 0.0
    macro_f1 = float(f1_score(y_true_np, y_pred_np, average="macro", zero_division=0)) if len(y_true_np) else 0.0
    return EpochMetrics(loss=mean_loss, accuracy=acc, macro_f1=macro_f1), y_true_np, y_pred_np


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_confusion_matrix_csv(path: Path, matrix: np.ndarray, class_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred", *class_names])
        for class_name, row in zip(class_names, matrix.tolist()):
            writer.writerow([class_name, *row])


def evaluate_and_save(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
    out_dir: Path,
    split_name: str,
) -> dict:
    metrics, y_true, y_pred = run_epoch(model, loader, criterion, device, optimizer=None)

    labels = list(range(len(class_names)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    payload = {
        "split": split_name,
        "loss": metrics.loss,
        "accuracy": metrics.accuracy,
        "macro_f1": metrics.macro_f1,
        "num_samples": int(len(y_true)),
    }

    save_json(out_dir / f"{split_name}_metrics.json", payload)
    save_json(out_dir / f"{split_name}_classification_report.json", report)
    save_confusion_matrix_csv(out_dir / f"{split_name}_confusion_matrix.csv", cm, class_names)
    return payload


def load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    class_names: list[str],
    image_size: int,
    epoch: int,
    best_val_macro_f1: float,
    cnn_backbone: str,
    clip_length: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "image_size": image_size,
            "epoch": epoch,
            "best_val_macro_f1": best_val_macro_f1,
            "cnn_backbone": cnn_backbone,
            "clip_length": clip_length,
        },
        path,
    )
