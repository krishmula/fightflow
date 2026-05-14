import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image, UnidentifiedImageError
from pathlib import Path
from tqdm import tqdm
import os

# ── Import all shared utilities from the base module (DRY) ────────────────
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

# Constants
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


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

    def _apply_consistent_transform(self, images: list[Image.Image]) -> torch.Tensor:
        cfg = self.transform
        if not isinstance(cfg, dict):
            return torch.stack([cfg(img) for img in images])

        aug_cfg = cfg["aug_cfg"]
        image_size = cfg["size"]
        do_flip = random.random() < aug_cfg.get("random_horizontal_flip_p", 0.5)
        jitter_cfg = aug_cfg.get("color_jitter", {"brightness": 0.2, "contrast": 0.2, "saturation": 0.15})
        b_f = random.uniform(max(0, 1 - jitter_cfg["brightness"]), 1 + jitter_cfg["brightness"])
        c_f = random.uniform(max(0, 1 - jitter_cfg["contrast"]), 1 + jitter_cfg["contrast"])
        s_f = random.uniform(max(0, 1 - jitter_cfg["saturation"]), 1 + jitter_cfg["saturation"])
        rot_deg = aug_cfg.get("random_rotation", {"degrees": 15})["degrees"]
        aff_cfg = aug_cfg.get("random_affine", {"degrees": 0, "translate": [0.1, 0.1], "scale": [0.9, 1.1], "shear": 5})
        affine_deg = aff_cfg.get("degrees", 0)
        angle_deg = affine_deg if affine_deg else rot_deg
        angle = random.uniform(-angle_deg, angle_deg)
        tx = random.uniform(-aff_cfg["translate"][0] * image_size, aff_cfg["translate"][0] * image_size)
        ty = random.uniform(-aff_cfg["translate"][1] * image_size, aff_cfg["translate"][1] * image_size)
        scale = random.uniform(aff_cfg["scale"][0], aff_cfg["scale"][1])
        shear = random.uniform(-aff_cfg["shear"], aff_cfg["shear"])
        erase_cfg = aug_cfg.get("random_erasing", {"p": 0.3, "scale": [0.02, 0.1], "ratio": [0.3, 3.3]})
        eraser = transforms.RandomErasing(
            p=erase_cfg.get("p", 0.0),
            scale=erase_cfg.get("scale", [0.02, 0.1]),
            ratio=erase_cfg.get("ratio", [0.3, 3.3]),
        )

        raw_frames = []
        for img in images:
            img = F.resize(img, (image_size, image_size))
            if do_flip: img = F.hflip(img)
            img = F.adjust_brightness(img, b_f)
            img = F.adjust_contrast(img, c_f)
            img = F.adjust_saturation(img, s_f)
            img = F.affine(img, angle=angle, translate=(tx, ty), scale=scale, shear=shear)
            raw_frames.append(F.to_tensor(img))

        motion_cfg = aug_cfg.get("motion_emphasis", {"p": 0.0, "alpha": 0.7})
        if motion_cfg.get("p", 0.0) > 0 and random.random() < motion_cfg["p"]:
            clip = torch.stack(raw_frames)
            diffs = torch.zeros_like(clip)
            diffs[1:] = (clip[1:] - clip[:-1]).abs()
            max_val = diffs.max()
            if max_val > 0:
                mask = diffs / max_val
                alpha = float(motion_cfg.get("alpha", 0.7))
                clip = (clip * (1.0 + alpha * mask)).clamp(0.0, 1.0)
                raw_frames = [frame for frame in clip]

        frames = []
        for t in raw_frames:
            t = F.normalize(t, mean=cfg["norm_cfg"]["mean"], std=cfg["norm_cfg"]["std"])
            if erase_cfg.get("p", 0.0) > 0:
                t = eraser(t)
            frames.append(t)
        return torch.stack(frames)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        clip_dir = Path(row["clip_dir"])
        label = int(row["label"])
        fps = sorted([p for p in clip_dir.iterdir() if p.suffix.lower() in SUPPORTED_IMAGE_EXTS])
        if not fps: raise RuntimeError(f"No frames in: {clip_dir}")
        idx = select_frame_indices(len(fps), self.clip_length)
        imgs = []
        for i in idx:
            try: imgs.append(Image.open(fps[i]).convert("RGB"))
            except: raise RuntimeError(f"Error reading: {fps[i]}")
        if self.transform: clip = self._apply_consistent_transform(imgs)
        else: clip = torch.stack([F.to_tensor(F.resize(img, (224, 224))) for img in imgs])
        return clip, label


def run_epoch(loader, model, criterion, device, optimizer=None, max_grad_norm=1.0):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    losses, y_t, y_p = [], [], []
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if is_train else None
    progress = tqdm(loader, leave=False, unit="batch")
    for clips, targets in progress:
        clips, targets = clips.to(device), targets.to(device)
        if is_train: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(clips)
                loss = criterion(logits, targets)
        if is_train:
            if scaler:
                scaler.scale(loss).backward()
                if max_grad_norm: scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                if max_grad_norm: nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
        losses.append(float(loss.item()))
        y_t.extend(targets.cpu().tolist())
        y_p.extend(torch.argmax(logits, 1).cpu().tolist())
        progress.set_postfix(loss=np.mean(losses[-10:]))
    from sklearn.metrics import accuracy_score, f1_score
    return EpochMetrics(float(np.mean(losses)), accuracy_score(y_t, y_p), f1_score(y_t, y_p, average="macro", zero_division=0)), np.array(y_t), np.array(y_p)



def save_checkpoint(path: Path, model: torch.nn.Module, **metadata) -> None:
    """Alias kept for legacy callers — delegates to base."""
    from ..base.utils import save_checkpoint as _sc
    _sc(path, model, **metadata)


def resolve_latest_checkpoint(backbone):
    base_dir = Path("runs") / backbone
    if not base_dir.exists(): return None
    runs = sorted([d for d in base_dir.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
    for run in runs:
        ckpt = run / "checkpoints" / "best.pt"
        if ckpt.exists(): return ckpt
    return None


def split_by_video(df, splits, seed=42):
    """
    Groups samples by video_id and assigns them to splits using GSGS.
    """
    if "split" in df.columns:
        if any(s in ["train", "val", "test"] for s in df["split"].dropna().unique()):
            return df[df["split"] == "train"], df[df["split"] == "val"], df[df["split"] == "test"]
    
    video_ids = sorted(df["video_id"].unique())
    if len(video_ids) < 2:
        return df, pd.DataFrame(), pd.DataFrame()

    # Count total samples per video
    video_totals = df.groupby("video_id").size().to_dict()
    # Count labels per video for stratification
    labels = sorted(df["punch_type"].unique().tolist())
    video_label_counts = df.groupby(["video_id", "punch_type"]).size().unstack(fill_value=0).to_dict('index')
    
    total_samples = len(df)
    train_ratio = splits.get("train", 0.8)
    val_ratio = splits.get("val", 0.2)
    test_ratio = splits.get("test", 0.0)
    total_ratio = train_ratio + val_ratio + test_ratio
    
    ratios = {
        "train": train_ratio / total_ratio,
        "val": val_ratio / total_ratio,
        "test": test_ratio / total_ratio
    }
    ratios = {k: v for k, v in ratios.items() if v > 0}
    
    targets_total = {k: total_samples * v for k, v in ratios.items()}
    
    rng = random.Random(seed)
    shuffled = video_ids[:]
    rng.shuffle(shuffled)
    shuffled.sort(key=lambda vid: (video_totals[vid], str(vid)), reverse=True)
    
    assignments = {k: [] for k in ratios}
    current_total = {k: 0.0 for k in ratios}
    
    def fill_ratio(name):
        return current_total[name] / targets_total[name] if targets_total[name] > 0 else float('inf')

    for vid in shuffled:
        vid_total = float(video_totals[vid])
        candidates = list(ratios.keys())
        # Prioritize fill_ratio to ensure 80/20 balance
        best_split = min(candidates, key=lambda s: (fill_ratio(s), current_total[s]))
        
        assignments[best_split].append(vid)
        current_total[best_split] += vid_total

    # Create split column
    vid_to_split = {}
    for split_name, vids in assignments.items():
        for v in vids:
            vid_to_split[v] = split_name
            
    df["split"] = df["video_id"].map(vid_to_split)
    
    df_t = df[df["split"] == "train"].copy()
    df_v = df[df["split"] == "val"].copy()
    df_te = df[df["split"] == "test"].copy()
    
    return df_t, df_v, df_te


def evaluate_and_save(model, loader, criterion, device, class_names, out_dir, split_name):
    # Perform a final clean evaluation
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


def get_transforms(image_size, augment=False, aug_cfg=None):
    if aug_cfg is None:
        aug_cfg = {}
    n_cfg = aug_cfg.get("normalize", {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]})
    if not augment:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=n_cfg["mean"], std=n_cfg["std"]),
        ])
    return {"size": image_size, "aug_cfg": aug_cfg, "norm_cfg": n_cfg}
