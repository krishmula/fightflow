import os
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import torchvision.transforms as T
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
from PIL import Image


def resolve_device(device_str: str) -> torch.device:
    """Resolve device string to torch.device."""
    if device_str == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Dict[str, Any]:
    """Load a PyTorch checkpoint."""
    return torch.load(checkpoint_path, map_location=device, weights_only=False)


def save_json(data: Dict[str, Any], path: Path) -> None:
    """Save dictionary as JSON."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def resolve_project_path(path_str: str) -> Path:
    """Resolve a path relative to the project root."""
    return Path(path_str)


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

    save_json(results, output_dir / f"{split_name}_results.json")
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