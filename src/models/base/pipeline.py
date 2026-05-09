#!/usr/bin/env python3
"""Train and evaluate a baseline CNN on folder-based frame classification data."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

DEFAULT_CLASS_NAMES = ["jab", "hook", "uppercut", "negative"]
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SUPPORTED_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (project_root() / path).resolve()


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    macro_f1: float


class FrameDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int, int | None, str]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample_path, label, frame_idx, _video_id = self.samples[index]
        suffix = sample_path.suffix.lower()

        if suffix in SUPPORTED_IMAGE_EXTS:
            try:
                image = Image.open(sample_path).convert("RGB")
            except (UnidentifiedImageError, OSError) as exc:
                raise RuntimeError(f"Could not read image: {sample_path}") from exc
        elif suffix in SUPPORTED_VIDEO_EXTS:
            image = self._read_video_frame(sample_path, frame_idx)
        else:
            raise RuntimeError(f"Unsupported file type in dataset: {sample_path}")

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def _read_video_frame(self, video_path: Path, frame_idx: int | None) -> Image.Image:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        try:
            if frame_idx is None:
                # Default to middle frame if no index provided
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_idx = frame_count // 2 if frame_count > 0 else 0
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            ok, frame_bgr = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                raise RuntimeError(f"Could not decode frame {frame_idx} from video: {video_path}")

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame_rgb)
        finally:
            cap.release()




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


import yaml

def get_data_config() -> dict:
    config_path = Path(__file__).parent / "data_config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_transforms(image_size: int):
    config = get_data_config()
    aug_cfg = config.get("augmentation", {})
    
    norm_cfg = aug_cfg.get("normalize", {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]})
    normalize = transforms.Normalize(mean=norm_cfg["mean"], std=norm_cfg["std"])
    
    # Basic augmentations
    flip_p = aug_cfg.get("random_horizontal_flip_p", 0.5)
    jitter_cfg = aug_cfg.get("color_jitter", {"brightness": 0.2, "contrast": 0.2, "saturation": 0.15})
    
    # Motion-specific augmentations
    rotation_cfg = aug_cfg.get("random_rotation", {"degrees": 15})
    affine_cfg = aug_cfg.get("random_affine", {"degrees": 0, "translate": [0.1, 0.1], "scale": [0.9, 1.1], "shear": 5})
    erasing_cfg = aug_cfg.get("random_erasing", {"p": 0.3, "scale": [0.02, 0.1], "ratio": [0.3, 3.3]})
    
    train_transforms = [
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=flip_p),
        transforms.ColorJitter(
            brightness=jitter_cfg["brightness"], 
            contrast=jitter_cfg["contrast"], 
            saturation=jitter_cfg["saturation"]
        ),
    ]
    
    # Add motion-specific augmentations
    if rotation_cfg["degrees"] > 0:
        train_transforms.append(transforms.RandomRotation(degrees=rotation_cfg["degrees"]))
    
    train_transforms.append(
        transforms.RandomAffine(
            degrees=affine_cfg["degrees"],
            translate=affine_cfg["translate"],
            scale=affine_cfg["scale"],
            shear=affine_cfg["shear"]
        )
    )
    
    train_transforms.extend([
        transforms.ToTensor(),
        normalize,
    ])
    
    # Add random erasing at the end (after ToTensor)
    if erasing_cfg["p"] > 0:
        train_transforms.append(
            transforms.RandomErasing(
                p=erasing_cfg["p"],
                scale=erasing_cfg["scale"],
                ratio=erasing_cfg["ratio"]
            )
        )
    
    train_tf = transforms.Compose(train_transforms)
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_tf, eval_tf


def collect_samples(class_names: list[str], frames_per_video: int = 1) -> list[tuple[Path, int, int | None, str]]:
    import pandas as pd
    
    config = get_data_config()
    
    csv_path = resolve_project_path(config.get("data", {}).get("annotations_file", "data/annotations.csv"))
    video_dir = resolve_project_path(config.get("data", {}).get("video_dir", "data/downloaded-videos"))
    uniform_samples_per_class = config.get("preparation", {}).get("uniform_samples_per_class")
    
    # Try to load from manifest first if it exists
    manifest_path = resolve_project_path(config.get("data", {}).get("manifest_file"))
    if manifest_path and manifest_path.exists():
        print(f"Loading samples from manifest: {manifest_path}")
        manifest_df = pd.read_csv(manifest_path)
        samples = []
        for _, row in manifest_df.iterrows():
            # Tuple format: (Path, label_idx, video_id, split_or_punch_type)
            # split_by_video uses index 2 for video_id now (fixed)
            samples.append((
                resolve_project_path(row["image_path"]),
                int(row["label"]),
                str(row["video_id"]),
                row.get("split", "train")
            ))
        return samples

    if not csv_path.exists():
        raise FileNotFoundError(f"Annotations CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    
    if uniform_samples_per_class == "auto":
        class_counts = df['punch_type'].astype(str).str.strip().str.lower().value_counts()
        target_counts = [class_counts.get(c.lower(), 0) for c in class_names]
        uniform_samples_per_class = min(target_counts) if target_counts else 0
        print(f"Auto-balancing classes. Setting uniform samples per class to: {uniform_samples_per_class}")

    if uniform_samples_per_class is not None:
        sampled_dfs = []
        for class_name in class_names:
            class_df = df[df['punch_type'].astype(str).str.strip().str.lower() == class_name.lower()]
            if class_df.empty:
                continue
            
            # Group by video_id and shuffle within each group
            grouped = [group.sample(frac=1, random_state=42).reset_index(drop=True) 
                       for _, group in class_df.groupby('video_id')]
            
            # Interleave frames from all videos to ensure perfect uniformity
            interleaved = []
            max_len = max((len(g) for g in grouped), default=0)
            for i in range(max_len):
                for g in grouped:
                    if i < len(g):
                        interleaved.append(g.iloc[i:i+1])
            
            if interleaved:
                class_sampled = pd.concat(interleaved, ignore_index=True)
                sampled_dfs.append(class_sampled.head(uniform_samples_per_class))
                
        if sampled_dfs:
            df = pd.concat(sampled_dfs, ignore_index=True)
    
    samples: list[tuple[Path, int, int | None, str]] = []
    
    for _, row in df.iterrows():
        video_id = row['video_id']
        frame_index = int(row['frame_index'])
        punch_type = str(row['punch_type']).strip().lower()
        
        label_idx = class_to_idx.get(punch_type, -1)
        if label_idx == -1:
            continue
            
        # Construct frame path
        video_stem = Path(video_id).stem
        frame_id = f"{punch_type}_{video_stem}_{frame_index:06d}"
        frames_dir = resolve_project_path(config.get("data", {}).get("processed_frames_dir", "data/processed/frames"))
        frame_path = frames_dir / f"{frame_id}.jpg"
        
        samples.append((frame_path, label_idx, video_id, punch_type))
    
    return samples


def split_by_video(
    samples: list[tuple[Path, int, int | None, str]], splits: dict, seed: int
) -> tuple[list[tuple[Path, int, int | None, str]], list[tuple[Path, int, int | None, str]], list[tuple[Path, int, int | None, str]]]:
    """
    Groups samples by video_id and assigns them to splits using a Group-aware Stratified Greedy Split (GSGS).
    This ensures that all frames from a single video stay in the same split (leakage-safe).
    """
    # Check if samples already have split assignments (from manifest)
    # Tuple format: (Path, label, video_id, split_or_punch_type)
    train_samples = [s for s in samples if s[3] == "train"]
    val_samples = [s for s in samples if s[3] == "val"]
    test_samples = [s for s in samples if s[3] == "test"]
    
    if train_samples or val_samples or test_samples:
        print(f"Using pre-assigned splits from manifest: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")
        return train_samples, val_samples, test_samples

    # Group samples by video_id
    video_to_samples = {}
    for s in samples:
        vid = s[2] # Use index 2 which is video_id
        if vid not in video_to_samples:
            video_to_samples[vid] = []
        video_to_samples[vid].append(s)

    video_ids = sorted(video_to_samples.keys())
    if len(video_ids) < 2:
        return samples, [], []

    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio <= 0:
        return samples, [], []

    split_ratios = {
        "train": train_ratio / total_ratio,
        "val": val_ratio / total_ratio,
        "test": test_ratio / total_ratio,
    }
    split_ratios = {k: v for k, v in split_ratios.items() if v > 0}

    # Count labels per video
    labels = sorted(list(set(s[1] for s in samples)))
    video_label_counts = {vid: {label: 0 for label in labels} for vid in video_ids}
    video_totals = {vid: 0 for vid in video_ids}
    
    for vid, v_samples in video_to_samples.items():
        for s in v_samples:
            label = s[1]
            video_label_counts[vid][label] += 1
            video_totals[vid] += 1

    total_samples = float(len(samples))
    targets_total = {k: total_samples * v for k, v in split_ratios.items()}
    
    label_totals = {label: 0 for label in labels}
    for s in samples:
        label = s[1]
        label_totals[label] += 1
        
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
    # Sort by size descending for greedy bin packing
    shuffled.sort(key=lambda vid: (video_totals[vid], str(vid)), reverse=True)

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
        vid_counts = video_label_counts[vid]
        vid_total = float(video_totals[vid])
        candidates = list(split_ratios.keys())
        best_split = min(
            candidates,
            key=lambda s: (cost(s, vid_counts, vid_total), fill_ratio(s), current_total[s]),
        )
        assignments[best_split].append(vid)
        current_total[best_split] += vid_total
        for label in labels:
            current_labels[label][best_split] += float(vid_counts.get(label, 0))

    def get_samples_for_vids(vids):
        res = []
        for vid in vids:
            res.extend(video_to_samples[vid])
        return res

    train_samples = get_samples_for_vids(assignments.get("train", []))
    val_samples = get_samples_for_vids(assignments.get("val", []))
    test_samples = get_samples_for_vids(assignments.get("test", []))
    
    return train_samples, val_samples, test_samples


def compute_class_weights(
    samples: list[tuple], num_classes: int, device: torch.device
) -> torch.Tensor | None:
    if not samples:
        return None

    labels = np.array([s[1] for s in samples], dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = np.zeros_like(counts)
    non_zero = counts > 0
    weights[non_zero] = counts.sum() / (num_classes * counts[non_zero])
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[EpochMetrics, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if is_train else None

    progress = tqdm(loader, leave=False, unit="batch")
    for inputs, targets in progress:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(inputs)
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
    save_confusion_matrix_plot(cm, class_names, out_dir / f"{split_name}_confusion_matrix.png")
    return payload


def make_dataloader(
    samples: list[tuple[Path, int, int | None]],
    transform,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    ds = FrameDataset(samples=samples, transform=transform)
    # MPS doesn't support pinned memory, so disable it for MPS devices
    pin_memory = not torch.backends.mps.is_available()
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def save_plots(history_rows: list[dict], output_dir: Path) -> None:
    """Generates training history plots for loss, accuracy, and F1 score."""
    if not history_rows:
        return
    
    epochs = [r["epoch"] for r in history_rows]
    
    # 1. Loss Plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [r["train_loss"] for r in history_rows], label="Train Loss", marker='o')
    plt.plot(epochs, [r["val_loss"] for r in history_rows], label="Val Loss", marker='o')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "loss_plot.png")
    plt.close()

    # 2. Accuracy Plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [r["train_acc"] for r in history_rows], label="Train Acc", marker='o')
    plt.plot(epochs, [r["val_acc"] for r in history_rows], label="Val Acc", marker='o')
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "accuracy_plot.png")
    plt.close()

    # 3. F1 Score Plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [r["train_macro_f1"] for r in history_rows], label="Train F1", marker='o')
    plt.plot(epochs, [r["val_macro_f1"] for r in history_rows], label="Val F1", marker='o')
    plt.title("Training and Validation Macro F1 Score")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "f1_score_plot.png")
    plt.close()


def save_confusion_matrix_plot(cm: np.ndarray, class_names: list[str], output_path: Path) -> None:
    """Generates a visual confusion matrix plot."""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize for text color
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_checkpoint(
    path: Path,
    model: nn.Module,
    class_names: list[str],
    image_size: int,
    epoch: int,
    best_val_macro_f1: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "image_size": image_size,
            "epoch": epoch,
            "best_val_macro_f1": best_val_macro_f1,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device)


def train(args: argparse.Namespace, model_class: type[nn.Module]) -> None:
    class_names = parse_class_names(args.class_names)

    set_seed(args.seed)
    if args.epochs < 1:
        raise SystemExit("--epochs must be >= 1")
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    splits = getattr(args, "splits", None)
    if not isinstance(splits, dict):
        raise ValueError("The 'splits' configuration is missing from the hparams file.")
    val_ratio = splits.get("val", 0.15)
    test_ratio = splits.get("test", 0.15)
    temp_ratio = val_ratio + test_ratio

    all_samples = collect_samples(class_names, frames_per_video=args.frames_per_video)
    if not all_samples:
        raise SystemExit("No samples found in annotations.csv")
    
    freeze_epochs = getattr(args, "freeze_epochs", 0)
        
    train_samples, val_samples, test_samples = split_by_video(all_samples, splits, args.seed)

    train_tf, eval_tf = get_transforms(args.image_size)
    train_loader = make_dataloader(
        train_samples,
        train_tf,
        args.batch_size,
        True,
        args.num_workers,
    )
    val_loader = make_dataloader(
        val_samples,
        eval_tf,
        args.batch_size,
        False,
        args.num_workers,
    )
    test_loader = (
        make_dataloader(
            test_samples,
            eval_tf,
            args.batch_size,
            False,
            args.num_workers,
        )
        if test_samples
        else None
    )

    # Instantiate model with optional attention parameter
    model_kwargs = {"num_classes": len(class_names)}
    if hasattr(args, 'use_attention'):
        model_kwargs["use_attention"] = args.use_attention
    model = model_class(**model_kwargs).to(device)
    class_weights = (
        compute_class_weights(train_samples, len(class_names), device) if args.use_class_weights else None
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Handle Backbone Freezing (Warmup Phase)
    if freeze_epochs > 0 and hasattr(model, "freeze_backbone"):
        print(f"Starting warmup phase: Freezing backbone for first {freeze_epochs} epochs.")
        model.freeze_backbone()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "image_size": args.image_size,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "device": str(device),
        "use_class_weights": bool(args.use_class_weights),
    }
    save_json(run_dir / "config.json", config_payload)

    best_macro_f1 = -1.0
    best_epoch = -1
    no_improve_epochs = 0
    history_rows: list[dict] = []

    print(
        f"Train samples: {len(train_samples)} | Val samples: {len(val_samples)} | "
        f"Test samples: {len(test_samples)}"
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
            )
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= args.patience:
            print(f"Early stopping triggered after {args.patience} epochs without val macro-F1 improvement.")
            break

        # Check if it's time to unfreeze
        if freeze_epochs > 0 and epoch == freeze_epochs and hasattr(model, "unfreeze_backbone"):
            print(f"\nWarmup phase complete. Unfreezing backbone for full fine-tuning.")
            model.unfreeze_backbone()
            # Update optimizer to include all parameters
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
            # Re-sync scheduler if needed? Or just let it continue. 
            # We'll re-sync to ensure it monitors the new loss landscape correctly.
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=args.lr_patience
            )

    history_path = run_dir / "history.csv"
    if history_rows:
        with history_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history_rows[0].keys()))
            writer.writeheader()
            writer.writerows(history_rows)
    
    # Save final plots
    save_plots(history_rows, run_dir)

    print(f"\nBest validation macro-F1: {best_macro_f1:.4f} at epoch {best_epoch}")

    best_ckpt = load_checkpoint(checkpoint_dir / "best.pt", device)
    model.load_state_dict(best_ckpt["model_state_dict"])

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

    if test_loader is not None:
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


def test_only(args: argparse.Namespace, model_class: type[nn.Module]) -> None:
    class_names_override = parse_class_names(args.class_names) if args.class_names else None

    device = resolve_device(args.device)
    checkpoint_path = resolve_project_path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, device)
    checkpoint_class_names = checkpoint.get("class_names", DEFAULT_CLASS_NAMES)
    if class_names_override is not None and class_names_override != checkpoint_class_names:
        raise SystemExit(
            "--class-names must match the class order used during training checkpoint creation."
        )
    class_names = checkpoint_class_names
    splits = getattr(args, "splits", None)
    if not isinstance(splits, dict):
        raise ValueError("The 'splits' configuration is missing from the hparams file.")
    val_ratio = splits.get("val", 0.15)
    test_ratio = splits.get("test", 0.15)
    temp_ratio = val_ratio + test_ratio
    seed = getattr(args, 'seed', 42)
    
    # Recreate the exact same split using identical seed
    all_samples = collect_samples(class_names, frames_per_video=args.frames_per_video)
    _, _, test_samples = split_by_video(all_samples, splits, seed)

    if not test_samples:
        raise SystemExit(f"No test images configured based on test_ratio: {test_ratio} or not found.")

    image_size = int(checkpoint.get("image_size", args.image_size))
    _, eval_tf = get_transforms(image_size)
    test_loader = make_dataloader(
        test_samples,
        eval_tf,
        args.batch_size,
        False,
        args.num_workers,
    )

    model = model_class(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.CrossEntropyLoss()

    report_root = resolve_project_path(args.report_dir)
    run_name = args.run_name or f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = report_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    test_summary = evaluate_and_save(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        class_names=class_names,
        out_dir=out_dir,
        split_name="test",
    )

    details = {
        "checkpoint": str(checkpoint_path),
        "class_names": class_names,
        "num_test_samples": len(test_samples),
        "device": str(device),
    }
    save_json(out_dir / "test_run_details.json", details)

    print("Test summary:", json.dumps(test_summary, indent=2))
    print(f"Saved test report to: {out_dir}")


def validate_only(args: argparse.Namespace, model_class: type[nn.Module]) -> None:
    class_names_override = parse_class_names(args.class_names) if args.class_names else None

    device = resolve_device(args.device)
    checkpoint_path = resolve_project_path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, device)
    checkpoint_class_names = checkpoint.get("class_names", DEFAULT_CLASS_NAMES)
    if class_names_override is not None and class_names_override != checkpoint_class_names:
        raise SystemExit(
            "--class-names must match the class order used during training checkpoint creation."
        )
    class_names = checkpoint_class_names

    splits = getattr(args, "splits", None)
    if not isinstance(splits, dict):
        raise ValueError("The 'splits' configuration is missing from the hparams file.")
    all_samples = collect_samples(class_names, frames_per_video=args.frames_per_video)
    if not all_samples:
        raise SystemExit(f"No samples found for classes: {class_names}")

    _, val_samples, _ = split_by_video(all_samples, splits, args.seed)

    if not val_samples:
        raise SystemExit("Not enough samples to construct a validation split.")

    image_size = int(checkpoint.get("image_size", args.image_size))
    _, eval_tf = get_transforms(image_size)
    val_loader = make_dataloader(
        val_samples,
        eval_tf,
        args.batch_size,
        False,
        args.num_workers,
    )

    model = model_class(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.CrossEntropyLoss()

    report_root = resolve_project_path(args.report_dir)
    run_name = args.run_name or f"val_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = report_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    val_summary = evaluate_and_save(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        class_names=class_names,
        out_dir=out_dir,
        split_name="val",
    )

    details = {
        "checkpoint": str(checkpoint_path),
        "class_names": class_names,
        "num_val_samples": len(val_samples),
        "device": str(device),
    }
    save_json(out_dir / "val_run_details.json", details)

    print("Validation summary:", json.dumps(val_summary, indent=2))
    print(f"Saved validation report to: {out_dir}")

