#!/usr/bin/env python3
"""Show train/val/test split composition by class and video before training.

Usage:
  # Base CNN models (cnn_baseline, cnn_advanced, resnet18, vgg_16)
  python scripts/check_split.py --model base

  # CNN-LSTM model
  python scripts/check_split.py --model cnn_lstm

Optional overrides (same defaults as training):
  --seed 42
  --splits train=0.70,val=0.15,test=0.15
  --class-names straight,hook,uppercut,none  # base only; defaults to classes in data_config.yaml
  --manifest data/processed/clip_manifest.csv  # cnn_lstm only

Comparing models:
  - Base CNN models (cnn_baseline, cnn_advanced, resnet18, vgg_16) all share the same
    split_by_video (GSGS) implementation as CNN-LSTM, so with an identical seed they
    receive the same leakage-safe video assignments. Their accuracy numbers are
    directly comparable only if both models use the same split ratios and seed.

  - CNN-LSTM operates on clip-level samples while base CNN models operate on
    frame-level samples, so even with identical video splits the sample counts will
    differ. Do not compare raw accuracy numbers directly across the two model families.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def parse_splits(raw: str) -> dict:
    result = {}
    for part in raw.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            result[k.strip()] = float(v.strip())
    return result


def print_split_table(split_name: str, class_names: list[str], video_class_counts: dict[str, dict[int, int]], label_to_name: dict[int, str]) -> None:
    """video_class_counts: {video_id -> {class_idx -> count}}"""
    total_per_class: dict[int, int] = defaultdict(int)
    for vid_counts in video_class_counts.values():
        for cls_idx, cnt in vid_counts.items():
            total_per_class[cls_idx] += cnt
    grand_total = sum(total_per_class.values())

    col_w = 10
    vid_w = 38
    sep = "-" * (vid_w + 3 + len(class_names) * (col_w + 3) + col_w + 1)

    print(f"\n{'='*len(sep)}")
    print(f"  {split_name.upper()}  —  {grand_total} samples across {len(video_class_counts)} video(s)")
    print(f"{'='*len(sep)}")

    header = f"{'Video':<{vid_w}} |"
    for name in class_names:
        header += f" {name[:col_w-1]:<{col_w-1}} |"
    header += f" {'Total':<{col_w-1}}"
    print(header)
    print(sep)

    for vid in sorted(video_class_counts.keys()):
        counts = video_class_counts[vid]
        row_total = sum(counts.values())
        row = f"{vid:<{vid_w}} |"
        for i in range(len(class_names)):
            row += f" {counts.get(i, 0):<{col_w-1}} |"
        row += f" {row_total:<{col_w-1}}"
        print(row)

    print(sep)
    totals_row = f"{'TOTAL':<{vid_w}} |"
    for i in range(len(class_names)):
        totals_row += f" {total_per_class.get(i, 0):<{col_w-1}} |"
    totals_row += f" {grand_total:<{col_w-1}}"
    print(totals_row)


def check_base(splits: dict, seed: int, class_names_arg: str | None) -> None:
    from src.models.base.pipeline import collect_samples, split_by_video, get_data_config

    if class_names_arg:
        class_names = [s.strip().lower() for s in class_names_arg.split(",") if s.strip()]
    else:
        class_names = get_data_config().get("classes", ["jab", "hook", "uppercut", "negative"])

    print(f"Model type : base CNN")
    print(f"Classes    : {class_names}")
    print(f"Splits     : {splits}")
    print(f"Seed       : {seed}")

    all_samples = collect_samples(class_names)
    if not all_samples:
        print("ERROR: No samples found. Check annotations.csv and video_dir in data_config.yaml.")
        return

    print(f"\nTotal annotated frames found: {len(all_samples)}")

    train_samples, val_samples, test_samples = split_by_video(all_samples, splits, seed)

    for split_name, samples in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        video_class_counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for path, label, frame_idx, video_id in samples:
            video_class_counts[video_id][label] += 1
        print_split_table(split_name, class_names, video_class_counts, {})


def check_cnn_lstm(splits: dict, seed: int, manifest_arg: str | None) -> None:
    from src.models.cnn_lstm.utils import get_data_config, load_manifest, resolve_project_path, split_by_video

    config = get_data_config()
    manifest_path = resolve_project_path(
        manifest_arg or config.get("data", {}).get("clip_manifest_file", "data/processed/clip_manifest.csv")
    )

    print(f"Model type : cnn_lstm")
    print(f"Manifest   : {manifest_path}")
    print(f"Splits     : {splits}")
    print(f"Seed       : {seed}")

    df = load_manifest(manifest_path)

    # Derive class names from label indices
    label_to_name: dict[int, str] = {}
    try:
        data_cfg = get_data_config()
        class_names = data_cfg.get("classes", [])
        if class_names:
            label_to_name = {i: name for i, name in enumerate(class_names)}
    except Exception:
        pass
    if not label_to_name:
        for lbl in sorted(df["label"].unique()):
            label_to_name[lbl] = str(lbl)

    class_names = [label_to_name[i] for i in range(len(label_to_name))]

    print(f"Classes    : {class_names}")
    print(f"\nTotal clips in manifest: {len(df)}")

    train_df, val_df, test_df = split_by_video(df, splits, seed)

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        video_class_counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for _, row in split_df.iterrows():
            video_class_counts[str(row["video_id"])][int(row["label"])] += 1
        print_split_table(split_name, class_names, video_class_counts, label_to_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect train/val/test split composition.")
    parser.add_argument("--model", choices=["base", "cnn_lstm"], default="base",
                        help="Which model pipeline to use (default: base)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--splits", default="train=0.70,val=0.15,test=0.15",
                        help="Split ratios, e.g. train=0.70,val=0.15,test=0.15")
    parser.add_argument("--class-names", default=None,
                        help="Comma-separated class names (base only). Default: jab,hook,uppercut,negative")
    parser.add_argument("--manifest", default=None,
                        help="Path to clip manifest CSV (cnn_lstm only)")
    args = parser.parse_args()

    splits = parse_splits(args.splits)

    if args.model == "base":
        check_base(splits, args.seed, args.class_names)
    else:
        check_cnn_lstm(splits, args.seed, args.manifest)


if __name__ == "__main__":
    main()
