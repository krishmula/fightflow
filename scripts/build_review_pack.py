#!/usr/bin/env python3
"""Build a human review pack for one video.

The review pack contains:
- a stratified sample queue from dataset_manifest.csv
- a padded MP4 clip for each sampled event
- a skeleton strip PNG for the matching skeleton_index
- a review CSV template keyed by sample_id / ann_row_index
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
os.environ.setdefault("MPLCONFIGDIR", str((_SCRIPTS.parent / ".matplotlib-cache").resolve()))

try:
    import cv2
except ImportError:
    cv2 = None

import audit_skeletons as sk_audit  # noqa: E402


REVIEW_COLUMNS = [
    "sample_id",
    "video_id",
    "ann_row_index",
    "class_clean",
    "rgb_label_match",
    "rgb_window_ok",
    "skeleton_match",
    "overall_decision",
    "notes",
]


def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["video_id"] = df["video_id"].astype(str).str.upper().str.strip()
    df["class_clean"] = df["class_clean"].astype(str).str.strip()
    return df


def load_validation(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["video_id"] = df["video_id"].astype(str).str.upper().str.strip()
    return df


def check_video_ready(video_id: str, validation_df: pd.DataFrame) -> None:
    row = validation_df.loc[validation_df["video_id"] == video_id]
    if row.empty:
        raise SystemExit(f"{video_id} missing from validation CSV")
    rec = row.iloc[0]
    if not bool(rec.get("has_video_file", False)):
        raise SystemExit(f"{video_id} has no video file according to validation CSV")
    if str(rec.get("download_status", "")).strip().lower() != "ok":
        raise SystemExit(f"{video_id} download_status is not ok")
    if not bool(rec.get("annotation_ranges_valid", False)):
        raise SystemExit(f"{video_id} annotation_ranges_valid is not True")


def filter_candidates(df: pd.DataFrame, video_id: str) -> pd.DataFrame:
    sub = df.loc[df["video_id"] == video_id].copy()
    if sub.empty:
        raise SystemExit(f"No manifest rows found for {video_id}")
    sub = sub.loc[
        (sub["quality_flag"].astype(str).str.strip().str.lower() == "ok")
        & (sub["has_video"].fillna(False).astype(bool))
        & (sub["has_skeleton"].fillna(False).astype(bool))
        & (sub["skeleton_row_aligned"].fillna(False).astype(bool))
    ].copy()
    if sub.empty:
        raise SystemExit(f"No reviewable manifest rows for {video_id} after filters")
    return sub


def stratified_sample(df: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    for class_name in sorted(df["class_clean"].dropna().unique()):
        grp = df.loc[df["class_clean"] == class_name].copy()
        if grp.empty:
            continue
        take = min(per_class, len(grp))
        chosen = rng.choice(grp.index.to_numpy(), size=take, replace=False)
        part = grp.loc[chosen].sort_values(["ann_row_index", "sample_id"])
        parts.append(part)
    if not parts:
        raise SystemExit("Stratified sample produced no rows")
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["class_clean", "ann_row_index", "sample_id"]).reset_index(drop=True)


def _clean_dir(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def write_rgb_clips(rows: pd.DataFrame, out_dir: Path, pad_frames: int) -> pd.DataFrame:
    if cv2 is None:
        raise SystemExit("OpenCV (cv2) is required; install opencv-python")
    video_paths = rows["source_video_path"].dropna().astype(str).str.strip().unique().tolist()
    if len(video_paths) != 1:
        raise SystemExit("Expected exactly one source video path in the sampled rows")
    video_path = Path(video_paths[0])
    if not video_path.is_file():
        raise SystemExit(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        raise SystemExit(f"Could not determine FPS for {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise SystemExit(f"Could not determine frame count for {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        raise SystemExit(f"Could not determine frame size for {video_path}")

    clip_cols = {"clip_start_frame": [], "clip_end_frame": [], "clip_path": []}
    _clean_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    try:
        for _, row in rows.iterrows():
            start = int(row["start_frame"])
            end = int(row["end_frame"])
            clip_start = max(0, start - pad_frames)
            clip_end = min(total_frames - 1, end + pad_frames)
            sample_id = str(row["sample_id"])
            ann_row = int(row["ann_row_index"])
            out_path = out_dir / f"{sample_id}_row{ann_row:04d}_clip.mp4"
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
            if not writer.isOpened():
                raise SystemExit(f"Could not open clip writer for {out_path}")
            try:
                for frame_idx in range(clip_start, clip_end + 1):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        raise SystemExit(f"Could not read frame {frame_idx} for {sample_id}")
                    writer.write(frame)
            finally:
                writer.release()
            if not out_path.is_file():
                raise SystemExit(f"Clip was not created: {out_path}")
            clip_cols["clip_start_frame"].append(clip_start)
            clip_cols["clip_end_frame"].append(clip_end)
            clip_cols["clip_path"].append(str(out_path.resolve()))
    finally:
        cap.release()
    return rows.assign(**clip_cols)


def write_skeleton_frames(rows: pd.DataFrame, out_dir: Path, max_frames: int) -> pd.DataFrame:
    cache: dict[str, np.ndarray] = {}
    _clean_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sk_pngs: list[str] = []
    for _, row in rows.iterrows():
        sk_path = str(row["skeleton_path"]).strip()
        if sk_path not in cache:
            path_obj = Path(sk_path)
            if not path_obj.is_file():
                raise SystemExit(f"Skeleton file not found: {path_obj}")
            cache[sk_path] = np.load(path_obj, mmap_mode="r")
        arr = cache[sk_path]
        idx = int(row["skeleton_index"])
        if idx < 0 or idx >= arr.shape[0]:
            raise SystemExit(f"skeleton_index {idx} out of range for {sk_path}")
        clip = np.asarray(arr[idx], dtype=np.float64)
        sample_id = str(row["sample_id"])
        ann_row = int(row["ann_row_index"])
        out_path = out_dir / f"{sample_id}_row{ann_row:04d}_skeleton.png"
        title = (
            f"{sample_id} — {row['class_clean']} — "
            f"ann_row={ann_row} sk_idx={idx}"
        )
        sk_audit.plot_skeleton_sequence(clip, out_path, title=title, max_frames=max_frames)
        sk_pngs.append(str(out_path.resolve()))
    return rows.assign(skeleton_png=sk_pngs)


def review_template(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.loc[:, ["sample_id", "video_id", "ann_row_index", "class_clean"]].copy()
    out["rgb_label_match"] = ""
    out["rgb_window_ok"] = ""
    out["skeleton_match"] = ""
    out["overall_decision"] = ""
    out["notes"] = ""
    return out.loc[:, REVIEW_COLUMNS]


def merge_existing_reviews(template: pd.DataFrame, review_path: Path) -> pd.DataFrame:
    if not review_path.is_file():
        return template
    template = template.copy()
    template["ann_row_index"] = template["ann_row_index"].astype(str)
    prev = pd.read_csv(review_path, dtype=str).fillna("")
    for col in REVIEW_COLUMNS:
        if col not in prev.columns:
            prev[col] = ""
    prev["ann_row_index"] = prev["ann_row_index"].astype(str)
    prev = prev.loc[:, REVIEW_COLUMNS]
    merged = template.merge(
        prev,
        on=["sample_id", "video_id", "ann_row_index", "class_clean"],
        how="left",
        suffixes=("", "_prev"),
    )
    for col in REVIEW_COLUMNS[4:]:
        prev_col = f"{col}_prev"
        merged[col] = merged[prev_col].fillna("")
        merged = merged.drop(columns=[prev_col])
    return merged.loc[:, REVIEW_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a human review pack for one video.")
    parser.add_argument("--video-id", type=str, required=True, help="e.g. V1")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/dataset_manifest.csv"),
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("data/processed/reports/video_annotation_validation.csv"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/reports/review_pack"),
    )
    parser.add_argument("--per-class", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pad-frames", type=int, default=8)
    parser.add_argument("--max-skeleton-frames", type=int, default=8)
    args = parser.parse_args()

    video_id = args.video_id.strip().upper()
    if not video_id:
        raise SystemExit("video_id is required")
    if args.per_class <= 0:
        raise SystemExit("--per-class must be > 0")
    if args.pad_frames < 0:
        raise SystemExit("--pad-frames must be >= 0")

    manifest = load_manifest(args.manifest)
    validation = load_validation(args.validation)
    check_video_ready(video_id, validation)
    candidates = filter_candidates(manifest, video_id)
    sample = stratified_sample(candidates, per_class=args.per_class, seed=args.seed)

    pack_dir = args.output_root / video_id
    rgb_dir = pack_dir / "rgb"
    clips_dir = pack_dir / "clips"
    skeleton_dir = pack_dir / "skeleton"
    queue_path = pack_dir / "review_queue.csv"
    review_path = pack_dir / "manual_review.csv"

    sample = sample.copy()
    _clean_dir(rgb_dir)
    sample = write_rgb_clips(sample, clips_dir, pad_frames=args.pad_frames)
    sample = write_skeleton_frames(sample, skeleton_dir, max_frames=args.max_skeleton_frames)

    queue_cols = [
        "sample_id",
        "video_id",
        "ann_row_index",
        "class_clean",
        "start_frame",
        "end_frame",
        "duration_frames",
        "source_video_path",
        "skeleton_path",
        "skeleton_index",
        "clip_start_frame",
        "clip_end_frame",
        "clip_path",
        "skeleton_png",
    ]
    pack_dir.mkdir(parents=True, exist_ok=True)
    sample.loc[:, queue_cols].to_csv(queue_path, index=False)

    template = review_template(sample)
    merged = merge_existing_reviews(template, review_path)
    merged.to_csv(review_path, index=False)

    counts = sample.groupby("class_clean").size().reset_index(name="n")
    print(f"Built review pack for {video_id}: {len(sample)} events")
    print(counts.to_string(index=False))
    print(f"Queue: {queue_path}")
    print(f"Manual review CSV: {review_path}")
    print(f"Clips: {clips_dir}")
    print(f"Skeleton PNGs: {skeleton_dir}")


if __name__ == "__main__":
    main()
