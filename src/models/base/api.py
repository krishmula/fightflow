import os
import random
import shutil
import cv2
import pandas as pd
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable
import yaml
import argparse

from .utils import resolve_project_path, save_json


class BaseModelAPI(ABC):
    """Base class for all model APIs providing common functionality."""

    def __init__(self):
        # Load the global data config
        config_path = Path(__file__).parent / "data_config.yaml"
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def train(self, args):
        """Train the model."""
        raise NotImplementedError("This base class cannot be trained directly.")

    def test(self, args):
        """Test the model."""
        raise NotImplementedError("This base class cannot be tested directly.")

    def prepare_data(self, args):
        """
        Prepare data based on model type:
        - Frame-only CNNs: extract single frames
        - CNN+LSTM: extract short clips around annotated frames
        """
        if isinstance(args, list):
            args = self._parse_args(args, mode="prepare_data", default_config=getattr(self, "default_config", None))

        model_name = getattr(self, "name", None)
        if model_name == "cnn_lstm":
            self._prepare_clips()
        else:
            self._prepare_frames(args)

    @staticmethod
    def _clean_args(raw_args: list[str]) -> list[str]:
        if raw_args and raw_args[0] == "--":
            return raw_args[1:]
        return raw_args

    @classmethod
    def _extract_config_path(cls, raw_args: list[str], default_config: Path = None) -> tuple[str | None, list[str]]:
        config_path: str | None = None
        filtered: list[str] = []
        i = 0
        while i < len(raw_args):
            token = raw_args[i]
            if token == "--config":
                if i + 1 >= len(raw_args):
                    raise SystemExit("--config requires a YAML file path")
                config_path = raw_args[i + 1]
                i += 2
                continue
            if token.startswith("--config="):
                config_path = token.split("=", 1)[1].strip()
                i += 1
                continue
            filtered.append(token)
            i += 1
            
        if config_path is None and default_config and default_config.exists():
            config_path = str(default_config)
                
        return config_path, filtered

    @staticmethod
    def _load_yaml(path_value: str) -> dict:
        cfg_path = resolve_project_path(path_value)
        if not cfg_path.is_file():
            raise SystemExit(f"Config file not found: {cfg_path}")

        with Path(cfg_path).open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}

        if not isinstance(payload, dict):
            raise SystemExit("Config YAML must contain a mapping at the top level")
        return payload

    @staticmethod
    def _select_mode_config(payload: dict, mode: str) -> dict:
        mode_keys = {"common", "train", "test", "validate"}
        if any(k in payload for k in mode_keys):
            out: dict = {}
            common = payload.get("common", {})
            if common is not None:
                if not isinstance(common, dict):
                    raise SystemExit("Config 'common' section must be a mapping")
                out.update(common)

            mode_cfg = payload.get(mode, {})
            if mode_cfg is not None:
                if not isinstance(mode_cfg, dict):
                    raise SystemExit(f"Config '{mode}' section must be a mapping")
                out.update(mode_cfg)
            return out

        return payload

    def _parse_args(self, raw_args: list[str], mode: str, default_config: Path = None) -> argparse.Namespace:
        config_dict = {}
        if default_config and default_config.exists():
            payload = self._load_yaml(str(default_config))
            config_dict = self._select_mode_config(payload, mode)
        return argparse.Namespace(**config_dict)

    def _invoke(self, mode: str, raw_args: list[str], runner: Callable, default_config: Path = None) -> int:
        parsed_args = self._parse_args(raw_args, mode, default_config)
        runner(parsed_args)
        return 0

    def _clear_dir(self, path: Path) -> None:
        if not path.exists():
            return
        for child in path.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    def _sample_annotations(self, df: pd.DataFrame, class_names: list[str]) -> pd.DataFrame:
        uniform_samples_per_class = self.config.get("preparation", {}).get("uniform_samples_per_class")

        if uniform_samples_per_class == "auto":
            class_counts = df["punch_type"].astype(str).str.strip().str.lower().value_counts()
            target_counts = [class_counts.get(c.lower(), 0) for c in class_names]
            uniform_samples_per_class = min(target_counts) if target_counts else 0
            print(f"Auto-balancing classes. Setting uniform samples per class to: {uniform_samples_per_class}")

        if uniform_samples_per_class is None:
            return df

        sampled_dfs = []
        for class_name in class_names:
            class_df = df[df["punch_type"].astype(str).str.strip().str.lower() == class_name.lower()]
            if class_df.empty:
                continue

            # Group by video_id and shuffle within each group
            grouped = [
                group.sample(frac=1, random_state=42).reset_index(drop=True)
                for _, group in class_df.groupby("video_id")
            ]

            # Interleave frames from all videos to ensure perfect uniformity
            interleaved = []
            max_len = max((len(g) for g in grouped), default=0)
            for i in range(max_len):
                for g in grouped:
                    if i < len(g):
                        interleaved.append(g.iloc[i : i + 1])

            if interleaved:
                class_sampled = pd.concat(interleaved, ignore_index=True)
                sampled_dfs.append(class_sampled.head(uniform_samples_per_class))

        if sampled_dfs:
            return pd.concat(sampled_dfs, ignore_index=True)
        return df

    def _prepare_frames(self, args: argparse.Namespace | None = None) -> None:
        annotations_file = resolve_project_path(self.config["data"]["annotations_file"])
        video_dir = resolve_project_path(self.config["data"]["video_dir"])
        frames_dir = resolve_project_path(self.config["data"]["processed_frames_dir"])
        manifest_file = resolve_project_path(self.config["data"]["manifest_file"])

        frames_dir.mkdir(parents=True, exist_ok=True)
        existing = list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.jpeg"))
        if existing:
            total_ext = len(existing)
            print(f"Clearing {total_ext} existing frames from {frames_dir}...")
            for i, f in enumerate(existing):
                f.unlink()
                if (i + 1) % 100 == 0 or (i + 1) == total_ext:
                    print(f"\r  Deleted: {i + 1}/{total_ext}", end="", flush=True)
            print()

        df = pd.read_csv(annotations_file)
        class_names = self.config["classes"]
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        print(f"Balancing dataset classes...")
        df = self._sample_annotations(df, class_names)
        
        split_ratios = getattr(args, "splits", None)
        if not isinstance(split_ratios, dict):
            raise ValueError("The 'splits' configuration is missing from the hparams file.")
        
        seed = getattr(args, "seed", None)
        if seed is None:
            raise ValueError("The 'seed' configuration is missing from the hparams file.")
        
        print(f"Applying Group-aware Stratified Greedy Split (GSGS) for leakage-safe preparation...")
        df = self._assign_splits_by_video(df, class_to_idx, split_ratios, seed)

        total_rows = len(df)
        samples = []
        skipped = 0
        skipped_rows = []

        print(f"\nStarting extraction: {total_rows} frames to process...")

        processed_count = 0
        for video_id, video_df in df.groupby("video_id", sort=False):
            video_path = video_dir / video_id
            
            if not video_path.exists():
                for _, row in video_df.iterrows():
                    processed_count += 1
                    skipped_rows.append({
                        "video_id": video_id,
                        "frame_index": int(row["frame_index"]),
                        "punch_type": str(row["punch_type"]),
                        "reason": "missing_video",
                    })
                    skipped += 1
                continue

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                for _, row in video_df.iterrows():
                    processed_count += 1
                    skipped_rows.append({
                        "video_id": video_id,
                        "frame_index": int(row["frame_index"]),
                        "punch_type": str(row["punch_type"]),
                        "reason": "video_open_failed",
                    })
                    skipped += 1
                continue

            # Sort by frame_index for better seek performance
            current_frame = -1
            for _, row in video_df.sort_values("frame_index").iterrows():
                processed_count += 1
                frame_index = int(row["frame_index"])
                punch_type = str(row["punch_type"])

                print(
                    f"\r  Progress: {processed_count}/{total_rows} | Extracted: {len(samples)} | Skipped: {skipped}",
                    end="",
                    flush=True,
                )

                # Performance Optimization: If the next frame is close, read sequentially instead of seeking
                diff = frame_index - current_frame
                if current_frame != -1 and 0 < diff < 100:
                    # Sequential read is often MUCH faster than seeking in many codecs
                    for _ in range(diff - 1):
                        cap.grab() # grab() skips decoding, making it faster
                    ret, frame = cap.read()
                else:
                    # Standard seek for large jumps
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ret, frame = cap.read()
                
                current_frame = frame_index

                if not ret:
                    skipped_rows.append({
                        "video_id": video_id,
                        "frame_index": frame_index,
                        "punch_type": punch_type,
                        "reason": "frame_decode_failed",
                    })
                    skipped += 1
                    continue

                label_idx = class_to_idx.get(punch_type, -1)
                if label_idx == -1:
                    skipped_rows.append({
                        "video_id": video_id,
                        "frame_index": frame_index,
                        "punch_type": punch_type,
                        "reason": "unknown_label",
                    })
                    skipped += 1
                    continue

                video_stem = Path(video_id).stem
                frame_filename = f"{punch_type}_{video_stem}_{frame_index:06d}.jpg"
                frame_path = frames_dir / frame_filename
                cv2.imwrite(str(frame_path), frame)

                samples.append({
                    "image_path": str(frame_path),
                    "label": label_idx,
                    "video_id": video_id,
                    "frame_index": frame_index,
                    "punch_type": punch_type,
                    "split": row.get("split", "train"),
                })
            cap.release()

        print()

        manifest_df = pd.DataFrame(samples)
        manifest_df.to_csv(manifest_file, index=False)

        if skipped_rows:
            skips_file = manifest_file.with_name("prepare_skips_frames.csv")
            pd.DataFrame(skipped_rows).to_csv(skips_file, index=False)

        print(f"Data preparation complete!")
        print(f"  - Extracted frames: {len(samples)}")
        print(f"  - Skipped rows:     {skipped}")
        print(f"  - Frames directory: {frames_dir}")
        print(f"  - Manifest file:    {manifest_file}")
        if skipped_rows:
            print(f"  - Skips log:        {skips_file}")

    def _assign_splits_by_video(
        self,
        df: pd.DataFrame,
        class_to_idx: dict[str, int],
        splits: dict,
        seed: int,
    ) -> pd.DataFrame:
        if df.empty:
            return df

        working = df.copy()
        working["label"] = (
            working["punch_type"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(class_to_idx)
            .fillna(-1)
            .astype(int)
        )
        working = working[working["label"] >= 0]
        if working.empty:
            df["split"] = "train"
            return df

        train_ratio = float(splits.get("train", 0.7))
        val_ratio = float(splits.get("val", 0.15))
        test_ratio = float(splits.get("test", 0.15))
        total_ratio = train_ratio + val_ratio + test_ratio
        if total_ratio <= 0:
            df["split"] = "train"
            return df

        split_ratios = {
            "train": train_ratio / total_ratio,
            "val": val_ratio / total_ratio,
            "test": test_ratio / total_ratio,
        }
        split_ratios = {k: v for k, v in split_ratios.items() if v > 0}

        video_ids = sorted(working["video_id"].astype(str).unique().tolist())
        if len(video_ids) < 2:
            df["split"] = "train"
            return df

        labels = sorted(working["label"].unique().tolist())
        video_label_counts = (
            working.groupby(["video_id", "label"]).size().unstack(fill_value=0).reindex(video_ids, fill_value=0)
        )
        video_totals = video_label_counts.sum(axis=1)

        total_samples = float(video_totals.sum())
        targets_total = {k: total_samples * v for k, v in split_ratios.items()}
        label_totals = working["label"].value_counts().to_dict()
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

        video_to_split = {
            vid: split for split, vids in assignments.items() for vid in vids
        }
        
        print(f"Leakage-safe split assignment complete (seed={seed}):")
        for split in ["train", "val", "test"]:
            if split in assignments:
                vids = assignments[split]
                count = int(current_total[split])
                print(f"  {split:5}: {count:5} samples from {len(vids):3} videos")

        df = df.copy()
        df["split"] = df["video_id"].map(video_to_split).fillna("train")
        return df

    def _prepare_clips(self) -> None:
        data_cfg = self.config.get("data", {})
        annotations_file = resolve_project_path(data_cfg["annotations_file"])
        video_dir = resolve_project_path(data_cfg["video_dir"])
        clips_dir = resolve_project_path(data_cfg.get("processed_clips_dir", "data/processed/clips"))
        manifest_file = resolve_project_path(data_cfg.get("clip_manifest_file", "data/processed/clip_manifest.csv"))

        clips_dir.mkdir(parents=True, exist_ok=True)
        if any(clips_dir.iterdir()):
            print(f"Clearing existing clips from {clips_dir} ...")
            self._clear_dir(clips_dir)
            clips_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(annotations_file)
        class_names = self.config["classes"]
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        print(f"Balancing dataset classes...")
        df = self._sample_annotations(df, class_names)
        
        # Add a dummy args object for compatibility with _assign_splits_by_video if needed
        # but cnn_lstm doesn't actually use the split column from the manifest yet.
        # However, for consistency, we could add it.
        
        total_rows = len(df)
        samples = []
        skipped = 0
        skipped_rows = []

        print(f"Starting clip extraction: {total_rows} annotations to process...")
        
        # Load splits and seed from the model's hparams.yaml ("common" section).
        hparams_common = {}
        default_config = getattr(self, "default_config", None)
        if default_config and Path(default_config).is_file():
            hparams_common = self._load_yaml(str(default_config)).get("common", {})
        splits = hparams_common.get("splits", {"train": 0.8, "val": 0.2, "test": 0.0})
        seed = hparams_common.get("seed", 42)
        df = self._assign_splits_by_video(df, class_to_idx, splits, seed)

        prep_cfg = self.config.get("preparation", {})
        pre_frames = int(prep_cfg.get("clip_pre_frames", 10))
        post_frames = int(prep_cfg.get("clip_post_frames", 5))
        clip_len = pre_frames + post_frames + 1

        processed_count = 0
        for video_id, video_df in df.groupby("video_id", sort=False):
            video_path = video_dir / video_id
            
            if not video_path.exists():
                for _, row in video_df.iterrows():
                    processed_count += 1
                    skipped_rows.append({
                        "video_id": video_id,
                        "frame_index": int(row["frame_index"]),
                        "punch_type": str(row["punch_type"]),
                        "reason": "missing_video",
                    })
                    skipped += 1
                continue

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                for _, row in video_df.iterrows():
                    processed_count += 1
                    skipped_rows.append({
                        "video_id": video_id,
                        "frame_index": int(row["frame_index"]),
                        "punch_type": str(row["punch_type"]),
                        "reason": "video_open_failed",
                    })
                    skipped += 1
                continue

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0:
                for _, row in video_df.iterrows():
                    processed_count += 1
                    skipped_rows.append({
                        "video_id": video_id,
                        "frame_index": int(row["frame_index"]),
                        "punch_type": str(row["punch_type"]),
                        "reason": "empty_video",
                    })
                    skipped += 1
                cap.release()
                continue

            # Sort by frame_index for better seek performance
            for _, row in video_df.sort_values("frame_index").iterrows():
                processed_count += 1
                frame_index = int(row["frame_index"])
                punch_type = str(row["punch_type"])

                print(
                    f"\r  Progress: {processed_count}/{total_rows} | Extracted: {len(samples)} | Skipped: {skipped}",
                    end="",
                    flush=True,
                )

                label_idx = class_to_idx.get(punch_type, -1)
                if label_idx == -1:
                    skipped_rows.append({
                        "video_id": video_id,
                        "frame_index": frame_index,
                        "punch_type": punch_type,
                        "reason": "unknown_label",
                    })
                    skipped += 1
                    continue

                video_stem = Path(video_id).stem
                # Use processed_count for unique clip_id if needed, or row index
                clip_id = f"{punch_type}_{video_stem}_{frame_index:06d}_v{processed_count:06d}"
                clip_path = clips_dir / clip_id
                clip_path.mkdir(parents=True, exist_ok=True)

                indices = [frame_index - pre_frames + offset for offset in range(clip_len)]
                success = True
                
                # Performance Optimization: Maintain current frame position to avoid redundant seeks
                current_cap_frame = -1
                for j, idx in enumerate(indices):
                    safe_idx = max(0, min(frame_count - 1, idx))
                    
                    if current_cap_frame != -1 and safe_idx == current_cap_frame + 1:
                        # Direct read is MUCH faster than seeking
                        ok, frame = cap.read()
                    else:
                        # Seek only when necessary
                        cap.set(cv2.CAP_PROP_POS_FRAMES, safe_idx)
                        ok, frame = cap.read()
                    
                    current_cap_frame = safe_idx
                    
                    if not ok or frame is None:
                        success = False
                        break
                    frame_path = clip_path / f"frame_{j:04d}.jpg"
                    if not cv2.imwrite(str(frame_path), frame):
                        success = False
                        break

                if not success:
                    skipped_rows.append({
                        "video_id": video_id,
                        "frame_index": frame_index,
                        "punch_type": punch_type,
                        "reason": "clip_extraction_failed",
                    })
                    shutil.rmtree(clip_path, ignore_errors=True)
                    skipped += 1
                    continue

                samples.append({
                    "clip_dir": str(clip_path),
                    "label": label_idx,
                    "video_id": video_id,
                    "center_frame": frame_index,
                    "clip_start": frame_index - pre_frames,
                    "clip_end": frame_index + post_frames,
                    "clip_length": clip_len,
                    "punch_type": punch_type,
                    "split": row["split"],
                })
            cap.release()

        print()

        manifest_df = pd.DataFrame(samples)
        manifest_df.to_csv(manifest_file, index=False)

        if skipped_rows:
            skips_file = manifest_file.with_name("prepare_skips_clips.csv")
            pd.DataFrame(skipped_rows).to_csv(skips_file, index=False)

        print(f"Data preparation complete!")
        print(f"  - Extracted clips:  {len(samples)}")
        print(f"  - Skipped rows:     {skipped}")
        print(f"  - Clips directory:  {clips_dir}")
        print(f"  - Manifest file:    {manifest_file}")
        if skipped_rows:
            print(f"  - Skips log:        {skips_file}")
