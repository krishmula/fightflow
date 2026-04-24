#!/usr/bin/env python3
"""Step 6: build unified dataset_manifest.csv (one row per usable annotation event)."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import audit_annotations as ann  # noqa: E402

# Known spreadsheet variants -> canonical label
CLASS_CLEAN_OVERRIDES: dict[str, str] = {
    "cross": "Cross",
    "jab": "Jab",
    "lead hook": "Lead Hook",
    "rear hook": "Rear Hook",
    "lead uppercut": "Lead Uppercut",
    "rear uppercut": "Rear Uppercut",
}


def clean_class(raw: str) -> str:
    s = " ".join(str(raw).strip().split())
    if not s:
        return s
    k = s.lower()
    if k in CLASS_CLEAN_OVERRIDES:
        return CLASS_CLEAN_OVERRIDES[k]
    return " ".join(w[:1].upper() + w[1:].lower() if w else w for w in s.split())


def load_skeleton_counts(report_path: Path) -> dict[str, tuple[int, bool]]:
    """video_id -> (num_events_dim0, matches_annotation_count_from_audit)."""
    if not report_path.is_file():
        return {}
    df = pd.read_csv(report_path)
    out: dict[str, tuple[int, bool]] = {}
    for _, r in df.iterrows():
        vid = str(r["video_id"]).strip().upper()
        n = int(r["num_events_dim0"]) if pd.notna(r.get("num_events_dim0")) else -1
        match = str(r.get("event_count_matches_annotation", "")).strip().lower() == "yes"
        out[vid] = (n, match)
    return out


def load_video_paths(report_path: Path, fallback_dir: Path) -> dict[str, tuple[str, str, bool]]:
    """video_id -> (resolved file_path, download_status, file_exists)."""
    if not report_path.is_file():
        return {}
    df = pd.read_csv(report_path)
    out: dict[str, tuple[str, str, bool]] = {}
    for _, r in df.iterrows():
        vid = str(r["video_id"]).strip().upper()
        fp = str(r.get("file_path", "") or "").strip()
        st = str(r.get("download_status", "") or "")
        resolved = ""
        exists = False
        for candidate in ([Path(fp)] if fp else []) + [fallback_dir / f"{vid}.mp4"]:
            if candidate and candidate.is_file():
                resolved = str(candidate.resolve())
                exists = True
                break
        out[vid] = (resolved, st, exists)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 6: build dataset_manifest.csv")
    parser.add_argument("--annotation-dir", type=Path, default=Path("data/raw/Annotation_files"))
    parser.add_argument(
        "--video-report",
        type=Path,
        default=Path("data/processed/reports/video_downloads.csv"),
    )
    parser.add_argument(
        "--skeleton-report",
        type=Path,
        default=Path("data/processed/reports/skeleton_audit_summary.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/dataset_manifest.csv"),
    )
    parser.add_argument(
        "--include-bad",
        action="store_true",
        help="Include rows that failed audit quality (is_bad); default excludes them",
    )
    args = parser.parse_args()

    ann_dir = args.annotation_dir
    video_info = load_video_paths(args.video_report, Path("data/downloaded-videos"))
    sk_info = load_skeleton_counts(args.skeleton_report)

    raw_root = Path("data/raw/Skeleton_data")

    rows: list[dict] = []

    for path in sorted(ann_dir.glob("V*.xlsx")):
        vid = ann._video_id(path)
        events, _ = ann.load_annotation_events(path)
        events = events.reset_index(drop=True)
        events_q = ann.annotate_quality(events)

        events_q.insert(0, "video_id", vid)
        events_q.insert(1, "source_file", path.name)
        events_q.insert(2, "ann_row_index", np.arange(len(events_q), dtype=int))

        total_ann_rows = len(events_q)
        if not args.include_bad:
            sub = events_q[~events_q["is_bad"]].copy()
        else:
            sub = events_q.copy()

        vpaths = video_info.get(vid, ("", "", False))
        vid_fp, vid_status, vid_exists = vpaths
        has_video = bool(vid_exists and vid_status == "ok")
        vid_fp_out = vid_fp if has_video else ""

        sk_path = raw_root / f"{vid}.npy"
        has_sk = sk_path.is_file()
        sk_tuple = sk_info.get(vid)
        if sk_tuple is not None:
            sk_n, _sk_match_audit = sk_tuple
        elif has_sk:
            arr = np.load(sk_path, mmap_mode="r")
            sk_n = int(arr.shape[0])
        else:
            sk_n = -1
        # One clip per spreadsheet row: skeleton dim0 must match total .xlsx rows (incl. bad rows).
        skeleton_row_aligned = sk_n == total_ann_rows and has_sk

        seq = 0
        for _, r in sub.iterrows():
            seq += 1
            sample_id = f"{vid}_{seq:05d}"
            cls_raw = r["class_raw"]
            cls_clean = clean_class(cls_raw)

            if vid == "V6":
                qflag = "review"
                note = "V6_skeleton_not_1to1_event_clips_alignment_TBD_step7"
            elif not skeleton_row_aligned:
                qflag = "review"
                note = "skeleton_dim0_mismatch_vs_annotation_row_count"
            else:
                qflag = "ok"
                note = ""

            rows.append(
                {
                    "sample_id": sample_id,
                    "video_id": vid,
                    "source_video_path": vid_fp_out,
                    "ann_row_index": int(r["ann_row_index"]),
                    "start_frame": float(r["start_frame"]),
                    "end_frame": float(r["end_frame"]),
                    "duration_frames": float(r["duration_inclusive"]),
                    "class_raw": cls_raw,
                    "class_clean": cls_clean,
                    "class_id": "",  # filled after pass
                    "split": "pending",
                    "skeleton_path": str(sk_path) if has_sk else "",
                    "skeleton_index": int(r["ann_row_index"]),
                    "skeleton_row_aligned": skeleton_row_aligned,
                    "has_video": has_video,
                    "has_skeleton": has_sk,
                    "quality_flag": qflag,
                    "notes": note,
                }
            )

    if not rows:
        raise SystemExit("No manifest rows produced.")

    mf = pd.DataFrame(rows)
    # Refine class_clean with full title-case pass for remaining multi-word
    uniq = sorted(mf["class_clean"].unique())
    cmap = {c: i for i, c in enumerate(uniq)}
    mf["class_id"] = mf["class_clean"].map(cmap)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mf.to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(mf)} rows)")


if __name__ == "__main__":
    main()
