import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image, UnidentifiedImageError
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
import yaml
import json
import random
import os

# Constants
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


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
        angle = random.uniform(-rot_deg, rot_deg)
        aff_cfg = aug_cfg.get("random_affine", {"degrees": 0, "translate": [0.1, 0.1], "scale": [0.9, 1.1], "shear": 5})
        tx = random.uniform(-aff_cfg["translate"][0] * image_size, aff_cfg["translate"][0] * image_size)
        ty = random.uniform(-aff_cfg["translate"][1] * image_size, aff_cfg["translate"][1] * image_size)
        scale = random.uniform(aff_cfg["scale"][0], aff_cfg["scale"][1])
        shear = random.uniform(-aff_cfg["shear"], aff_cfg["shear"])

        frames = []
        for img in images:
            img = F.resize(img, (image_size, image_size))
            if do_flip: img = F.hflip(img)
            img = F.adjust_brightness(img, b_f)
            img = F.adjust_contrast(img, c_f)
            img = F.adjust_saturation(img, s_f)
            img = F.affine(img, angle=angle, translate=(tx, ty), scale=scale, shear=shear)
            t = F.to_tensor(img)
            t = F.normalize(t, mean=cfg["norm_cfg"]["mean"], std=cfg["norm_cfg"]["std"])
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


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    macro_f1: float


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
    return EpochMetrics(float(np.mean(losses)), accuracy_score(y_t, y_p), f1_score(y_t, y_p, average="macro")), np.array(y_t), np.array(y_p)


def save_checkpoint(**kwargs):
    path = kwargs.pop("path")
    model = kwargs.pop("model")
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {"model_state_dict": model.state_dict()}
    ckpt.update(kwargs)
    torch.save(ckpt, path)


def load_checkpoint(path, device):
    return torch.load(path, map_location=device, weights_only=False)


def resolve_latest_checkpoint(backbone):
    base_dir = Path("runs") / backbone
    if not base_dir.exists(): return None
    runs = sorted([d for d in base_dir.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
    for run in runs:
        ckpt = run / "checkpoints" / "best.pt"
        if ckpt.exists(): return ckpt
    return None


def split_by_video(df, splits, seed=42):
    if "split" in df.columns:
        if any(s in ["train", "val", "test"] for s in df["split"].dropna().unique()):
            return df[df["split"] == "train"], df[df["split"] == "val"], df[df["split"] == "test"]
    v_ids = sorted(df["video_id"].unique())
    random.Random(seed).shuffle(v_ids)
    n = len(v_ids)
    t_e = int(n * splits.get("train", 0.7))
    v_e = t_e + int(n * splits.get("val", 0.15))
    t_ids, v_ids_s, te_ids = set(v_ids[:t_e]), set(v_ids[t_e:v_e]), set(v_ids[v_e:])
    df_t, df_v, df_te = df[df["video_id"].isin(t_ids)].copy(), df[df["video_id"].isin(v_ids_s)].copy(), df[df["video_id"].isin(te_ids)].copy()
    for d, s in zip([df_t, df_v, df_te], ["train", "val", "test"]): d["split"] = s
    return df_t, df_v, df_te


def evaluate_and_save(model, loader, criterion, device, class_names, out_dir, split_name):
    # Perform a final clean evaluation
    metrics, y_true, y_pred = run_epoch(loader, model, criterion, device)
    
    from sklearn.metrics import classification_report, confusion_matrix
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = {
        "metrics": {"loss": metrics.loss, "accuracy": metrics.accuracy, "macro_f1": metrics.macro_f1},
        "classification_report": classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    }
    save_json(out_dir / f"{split_name}_metrics.json", rep)
    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(out_dir / f"{split_name}_confusion_matrix.csv")
    return rep["metrics"]


def get_transforms(image_size, augment=False, aug_cfg=None):
    if aug_cfg is None: aug_cfg = {}
    n_cfg = aug_cfg.get("normalize", {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]})
    if not augment: return transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(), transforms.Normalize(mean=n_cfg["mean"], std=n_cfg["std"])])
    return {"size": image_size, "aug_cfg": aug_cfg, "norm_cfg": n_cfg}
