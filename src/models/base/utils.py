import os
import json
import random
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import torchvision.transforms as T
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
from PIL import Image


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_str: str = "auto") -> torch.device:
    """Resolve device string to torch.device (cuda > mps > cpu)."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Dict[str, Any]:
    """Load a PyTorch checkpoint."""
    return torch.load(checkpoint_path, map_location=device, weights_only=False)


def save_json(path: Path, data: Dict[str, Any], indent: int = 4) -> None:
    """Save dictionary as JSON. Canonical signature: (path, data)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)


def resolve_project_path(path: Any) -> Path:
    """Resolve a path string or Path to an absolute Path."""
    return Path(path).resolve()


def get_data_config(config_path: Optional[Path] = None) -> dict:
    """Load the shared data_config.yaml from the base models directory."""
    if config_path is None:
        config_path = Path(__file__).parent / "data_config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_yaml(path: Any) -> dict:
    """Load any YAML file and return as a dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_class_names(class_names: str) -> List[str]:
    """Split a comma-separated class name string into a list."""
    return [c.strip() for c in class_names.split(",")]


def load_manifest(manifest_path: Any) -> pd.DataFrame:
    """Load a CSV manifest file, raising clearly if it is missing."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    return pd.read_csv(manifest_path)


def compute_class_weights(
    labels: np.ndarray, num_classes: int, device: torch.device
) -> Optional[torch.Tensor]:
    """Compute inverse-frequency class weights for CrossEntropyLoss."""
    if len(labels) == 0:
        return None
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = np.zeros_like(counts)
    non_zero = counts > 0
    weights[non_zero] = counts.sum() / (num_classes * counts[non_zero])
    return torch.tensor(weights, dtype=torch.float32, device=device)


def save_checkpoint(path: Path, model: torch.nn.Module, **metadata) -> None:
    """Save a model checkpoint with optional metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {"model_state_dict": model.state_dict()}
    ckpt.update(metadata)
    torch.save(ckpt, path)


@dataclass
class EpochMetrics:
    """Aggregated metrics for a single training/validation epoch."""
    loss: float
    accuracy: float
    macro_f1: float


def split_by_video(df: pd.DataFrame, splits: dict, seed: int = 42):
    """
    Group-aware Stratified Greedy Split (GSGS): assigns whole videos to
    train/val/test so that no video leaks across splits.
    """
    if "split" in df.columns:
        if any(s in ["train", "val", "test"] for s in df["split"].dropna().unique()):
            return (
                df[df["split"] == "train"],
                df[df["split"] == "val"],
                df[df["split"] == "test"],
            )

    video_ids = sorted(df["video_id"].unique())
    if len(video_ids) < 2:
        return df, pd.DataFrame(), pd.DataFrame()

    video_totals = df.groupby("video_id").size().to_dict()
    total_samples = len(df)

    train_r = splits.get("train", 0.8)
    val_r   = splits.get("val",   0.2)
    test_r  = splits.get("test",  0.0)
    total_r = train_r + val_r + test_r
    ratios  = {k: v / total_r for k, v in
               [("train", train_r), ("val", val_r), ("test", test_r)] if v > 0}

    targets = {k: total_samples * v for k, v in ratios.items()}

    rng = random.Random(seed)
    shuffled = video_ids[:]
    rng.shuffle(shuffled)
    shuffled.sort(key=lambda vid: (video_totals[vid], str(vid)), reverse=True)

    assignments   = {k: [] for k in ratios}
    current_total = {k: 0.0 for k in ratios}

    def fill_ratio(name: str) -> float:
        return current_total[name] / targets[name] if targets[name] > 0 else float("inf")

    for vid in shuffled:
        best = min(ratios, key=lambda s: (fill_ratio(s), current_total[s]))
        assignments[best].append(vid)
        current_total[best] += float(video_totals[vid])

    vid_to_split = {v: s for s, vids in assignments.items() for v in vids}
    df = df.copy()
    df["split"] = df["video_id"].map(vid_to_split)

    return (
        df[df["split"] == "train"].copy(),
        df[df["split"] == "val"].copy(),
        df[df["split"] == "test"].copy(),
    )


def get_transforms(image_size: int) -> Tuple[T.Compose, T.Compose]:
    """Get train and eval transforms."""
    train_tf = T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


def make_dataloader(samples: List[Tuple[Path, int]], transform: T.Compose, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    """Create a DataLoader from samples."""
    dataset = FrameDataset(samples, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def evaluate_and_save(model: nn.Module, dataloader: DataLoader, class_names: List[str], device: torch.device, output_dir: Path, split_name: str) -> Dict[str, Any]:
    """Evaluate model and save results."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)

    # Save results
    results = {
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "macro_f1": report["macro avg"]["f1-score"],
        "accuracy": report["accuracy"],
    }

    save_json(output_dir / f"{split_name}_results.json", results)
    return results


class FrameDataset(torch.utils.data.Dataset):
    """Dataset for loading pre-extracted frames."""

    def __init__(self, samples: List[Tuple[Path, int]], transform: Optional[T.Compose] = None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label