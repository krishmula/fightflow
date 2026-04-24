#!/usr/bin/env python3
"""Step 5: verify annotation frame ranges fit each downloaded video.

Reads video metadata from video_downloads.csv and annotations via audit_annotations.
Optional: write spot-check PNGs from OpenCV frame grabs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import audit_annotations as ann  # noqa: E402

try:
    import cv2
except ImportError:
    cv2 = None


def load_video_report(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["video_id"] = df["video_id"].astype(str).str.upper().str.strip()
    return df


def max_end_frame(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    return float(valid.max()) if len(valid) else float("nan")


def spotcheck_events(
    video_path: Path,
    events_ok: pd.DataFrame,
    out_dir: Path,
    n: int,
    seed: int | None,
) -> None:
    if cv2 is None:
        raise SystemExit("OpenCV (cv2) required for --spotcheck; pip install opencv-python")
    if not video_path.is_file():
        raise SystemExit(f"Video not found: {video_path}")
    take = min(n, len(events_ok))
    if take == 0:
        return
    rows = events_ok.sample(n=take, random_state=seed) if seed is not None else events_ok.sample(n=take)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    try:
        for _, row in rows.iterrows():
            s = int(row["start_frame"])
            e = int(row["end_frame"])
            mid = (s + e) // 2
            idx = int(row["row_index"])
            for label, fi in (("start", s), ("mid", mid), ("end", e)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                name = f"{row['video_id']}_row{idx:04d}_{label}_f{fi}.png"
                cv2.imwrite(str(out_dir / name), frame)
    finally:
        cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 5: validate annotations vs video frame counts.")
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=Path("data/raw/Annotation_files"),
    )
    parser.add_argument(
        "--video-report",
        type=Path,
        default=Path("data/processed/reports/video_downloads.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/reports/video_annotation_validation.csv"),
    )
    parser.add_argument("--spotcheck-video", type=str, default="", help="e.g. V1")
    parser.add_argument("--spotcheck-n", type=int, default=3)
    parser.add_argument("--spotcheck-dir", type=Path, default=Path("data/processed/reports/spotcheck"))
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    ann_dir = args.annotation_dir
    report_path = args.video_report
    if not report_path.is_file():
        raise SystemExit(f"Missing {report_path} (run download_videos.py first)")

    vdf = load_video_report(report_path)
    vid_to_row = {r["video_id"]: r for _, r in vdf.iterrows()}

    xlsx_files = sorted(ann_dir.glob("V*.xlsx"))
    if not xlsx_files:
        raise SystemExit(f"No V*.xlsx under {ann_dir}")

    out_rows: list[dict] = []

    for path in xlsx_files:
        vid = ann._video_id(path)
        events, _ = ann.load_annotation_events(path)
        events = events.reset_index(drop=True)
        events_q = ann.annotate_quality(events)
        events_q.insert(0, "video_id", vid)
        events_q.insert(1, "row_index", np.arange(len(events_q), dtype=int))

        ok = events_q[~events_q["is_bad"]]
        max_ok = max_end_frame(ok["end_frame"]) if len(ok) else float("nan")
        max_all = max_end_frame(events_q["end_frame"])

        vr = vid_to_row.get(vid)
        if vr is None:
            fp = ""
            nfs = np.nan
            fps = np.nan
            dst = ""
            has_file = False
        else:
            fp = str(vr.get("file_path", "") or "")
            p = Path(fp)
            has_file = p.is_file()
            try:
                nfs = int(float(vr["num_frames"])) if pd.notna(vr.get("num_frames")) else float("nan")
            except (TypeError, ValueError):
                nfs = float("nan")
            try:
                fps = float(vr["fps"]) if pd.notna(vr.get("fps")) else float("nan")
            except (TypeError, ValueError):
                fps = float("nan")
            dst = str(vr.get("download_status", "") or "")

        notes: list[str] = []
        if vr is None:
            notes.append("video_id_missing_in_video_downloads_csv")
        elif dst != "ok":
            notes.append(f"download_status_{dst}")
        if fp and not has_file:
            notes.append("file_path_not_on_disk")

        valid = False
        if has_file and np.isfinite(nfs) and np.isfinite(max_ok):
            valid = max_ok <= nfs - 1
        elif not has_file or not np.isfinite(nfs):
            valid = False
            if not notes:
                notes.append("cannot_validate_missing_metadata")

        if np.isfinite(max_all) and np.isfinite(nfs) and has_file:
            if max_all > nfs - 1:
                notes.append("max_end_includes_bad_rows_exceeds_video")

        out_rows.append(
            {
                "video_id": vid,
                "has_video_file": has_file,
                "download_status": dst,
                "fps": fps if np.isfinite(fps) else "",
                "num_frames": int(nfs) if np.isfinite(nfs) else "",
                "max_end_frame_ok_rows": max_ok if np.isfinite(max_ok) else "",
                "max_end_frame_all_rows": max_all if np.isfinite(max_all) else "",
                "annotation_ranges_valid": valid,
                "time_unit": "frames",
                "notes": ";".join(notes),
            }
        )

        sc = args.spotcheck_video.strip().upper()
        if sc and sc == vid:
            vp = Path(fp) if fp else Path()
            if vp.is_file():
                spotcheck_events(vp, ok, args.spotcheck_dir, args.spotcheck_n, args.seed)
                print(f"Wrote spot-check images under {args.spotcheck_dir}")

    out_df = pd.DataFrame(out_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(out_df)} videos)")


if __name__ == "__main__":
    main()
