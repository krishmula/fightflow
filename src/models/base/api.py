import os
import shutil
import cv2
import pandas as pd
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable
import yaml
from sklearn.model_selection import train_test_split
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
        model_name = getattr(self, "name", None)
        if model_name == "cnn_lstm":
            self._prepare_clips()
        else:
            self._prepare_frames()

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

    def _invoke(self, mode: str, raw_args: list[str], runner: Callable, default_config: Path = None, defaults: dict = None) -> int:
        parsed_args = self._parse_args(raw_args, mode, default_config)
        config_dict = vars(parsed_args)

        if defaults:
            for k, v in defaults.items():
                if k not in config_dict:
                    config_dict[k] = v

        runner(argparse.Namespace(**config_dict))
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

    def _prepare_frames(self) -> None:
        annotations_file = resolve_project_path(self.config["data"]["annotations_file"])
        video_dir = resolve_project_path(self.config["data"]["video_dir"])
        frames_dir = resolve_project_path(self.config["data"]["processed_frames_dir"])
        manifest_file = resolve_project_path(self.config["data"]["manifest_file"])

        frames_dir.mkdir(parents=True, exist_ok=True)
        existing = list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.jpeg"))
        if existing:
            print(f"Clearing {len(existing)} existing frames from {frames_dir} ...")
            for f in existing:
                f.unlink()

        df = pd.read_csv(annotations_file)
        class_names = self.config["classes"]
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        df = self._sample_annotations(df, class_names)

        total_rows = len(df)
        samples = []
        skipped = 0
        skipped_rows = []

        print(f"Starting data preparation: {total_rows} annotations to process...")

        for i, (_, row) in enumerate(df.iterrows()):
            video_id = str(row["video_id"])
            punch_type = str(row["punch_type"])
            frame_index = int(row["frame_index"])

            print(
                f"\r  Progress: {i + 1}/{total_rows} | Extracted: {len(samples)} | Skipped: {skipped}",
                end="",
                flush=True,
            )

            video_path = video_dir / video_id
            if not video_path.exists():
                skipped_rows.append(
                    {
                        "video_id": video_id,
                        "frame_index": frame_index,
                        "punch_type": punch_type,
                        "reason": "missing_video",
                    }
                )
                skipped += 1
                continue

            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                skipped_rows.append(
                    {
                        "video_id": video_id,
                        "frame_index": frame_index,
                        "punch_type": punch_type,
                        "reason": "frame_decode_failed",
                    }
                )
                skipped += 1
                continue

            label_idx = class_to_idx.get(punch_type, -1)
            if label_idx == -1:
                skipped_rows.append(
                    {
                        "video_id": video_id,
                        "frame_index": frame_index,
                        "punch_type": punch_type,
                        "reason": "unknown_label",
                    }
                )
                skipped += 1
                continue

            video_stem = Path(video_id).stem
            frame_filename = f"{punch_type}_{video_stem}_{frame_index:06d}.jpg"
            frame_path = frames_dir / frame_filename
            cv2.imwrite(str(frame_path), frame)

            samples.append(
                {
                    "image_path": str(frame_path),
                    "label": label_idx,
                    "video_id": video_id,
                    "frame_index": frame_index,
                    "punch_type": punch_type,
                }
            )

        print()

        manifest_df = pd.DataFrame(samples)
        manifest_df.to_csv(manifest_file, index=False)

        if skipped_rows:
            skips_file = manifest_file.with_name("prepare_skips_frames.csv")
            pd.DataFrame(skipped_rows).to_csv(skips_file, index=False)

        print(
            f"Done! Extracted {len(samples)} samples  |  Skipped {skipped}  |  Saved to {frames_dir}"
        )
        print(f"Manifest: {manifest_file}")
        if skipped_rows:
            print(f"Skipped rows: {skips_file}")

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
        df = self._sample_annotations(df, class_names)

        prep_cfg = self.config.get("preparation", {})
        pre_frames = int(prep_cfg.get("clip_pre_frames", 10))
        post_frames = int(prep_cfg.get("clip_post_frames", 5))
        clip_len = pre_frames + post_frames + 1

        total_rows = len(df)
        samples = []
        skipped = 0
        skipped_rows = []

        print(f"Starting clip preparation: {total_rows} annotations to process...")

        for i, (_, row) in enumerate(df.iterrows()):
            video_id = str(row["video_id"])
            punch_type = str(row["punch_type"])
            frame_index = int(row["frame_index"])

            print(
                f"\r  Progress: {i + 1}/{total_rows} | Extracted: {len(samples)} | Skipped: {skipped}",
                end="",
                flush=True,
            )

            video_path = video_dir / video_id
            if not video_path.exists():
                skipped_rows.append(
                    {
                        "video_id": video_id,
                        "frame_index": frame_index,
                        "punch_type": punch_type,
                        "reason": "missing_video",
                    }
                )
                skipped += 1
                continue

            label_idx = class_to_idx.get(punch_type, -1)
            if label_idx == -1:
                skipped_rows.append(
                    {
                        "video_id": video_id,
                        "frame_index": frame_index,
                        "punch_type": punch_type,
                        "reason": "unknown_label",
                    }
                )
                skipped += 1
                continue

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                skipped_rows.append(
                    {
                        "video_id": video_id,
                        "frame_index": frame_index,
                        "punch_type": punch_type,
                        "reason": "video_open_failed",
                    }
                )
                skipped += 1
                continue

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0:
                skipped_rows.append(
                    {
                        "video_id": video_id,
                        "frame_index": frame_index,
                        "punch_type": punch_type,
                        "reason": "empty_video",
                    }
                )
                cap.release()
                skipped += 1
                continue

            video_stem = Path(video_id).stem
            clip_id = f"{punch_type}_{video_stem}_{frame_index:06d}_r{i + 1:06d}"
            clip_path = clips_dir / clip_id
            clip_path.mkdir(parents=True, exist_ok=True)

            indices = [frame_index - pre_frames + offset for offset in range(clip_len)]
            success = True
            for j, idx in enumerate(indices):
                safe_idx = max(0, min(frame_count - 1, idx))
                cap.set(cv2.CAP_PROP_POS_FRAMES, safe_idx)
                ok, frame = cap.read()
                if not ok or frame is None:
                    success = False
                    break
                frame_path = clip_path / f"frame_{j:04d}.jpg"
                if not cv2.imwrite(str(frame_path), frame):
                    success = False
                    break

            cap.release()

            if not success:
                skipped_rows.append(
                    {
                        "video_id": video_id,
                        "frame_index": frame_index,
                        "punch_type": punch_type,
                        "reason": "clip_write_failed",
                    }
                )
                shutil.rmtree(clip_path, ignore_errors=True)
                skipped += 1
                continue

            samples.append(
                {
                    "clip_dir": str(clip_path),
                    "label": label_idx,
                    "video_id": video_id,
                    "center_frame": frame_index,
                    "clip_start": frame_index - pre_frames,
                    "clip_end": frame_index + post_frames,
                    "clip_length": clip_len,
                    "punch_type": punch_type,
                }
            )

        print()

        manifest_df = pd.DataFrame(samples)
        manifest_df.to_csv(manifest_file, index=False)

        if skipped_rows:
            skips_file = manifest_file.with_name("prepare_skips_clips.csv")
            pd.DataFrame(skipped_rows).to_csv(skips_file, index=False)

        print(
            f"Done! Extracted {len(samples)} samples  |  Skipped {skipped}  |  Saved to {clips_dir}"
        )
        print(f"Manifest: {manifest_file}")
        if skipped_rows:
            print(f"Skipped rows: {skips_file}")
