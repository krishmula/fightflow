#!/usr/bin/env python3
"""Audit boxing annotation Excel files: per-video summary + event-level report.

Each raw workbook uses a slightly different layout; loaders below normalize to
start_frame, end_frame, class_raw. Duration is inclusive frame count:
    duration_inclusive = end_frame - start_frame + 1
(valid only when end >= start and both are finite).
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

# Flag events longer than this (frames) for review; not treated as "bad" by default.
DURATION_WARN_FRAMES = 500


def _video_id(path: Path) -> str:
    m = re.match(r"^(V\d+)$", path.stem, re.I)
    if not m:
        raise ValueError(f"Expected filename like V7.xlsx, got {path.name}")
    return m.group(1).upper()


def _strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def _load_v1_v10_style(path: Path, start: str, end: str, klass: str) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl", header=0)
    df = _strip_columns(df)
    sub = df[[start, end, klass]].rename(
        columns={start: "start_frame", end: "end_frame", klass: "class_raw"}
    )
    return sub


def _load_no_header_first3(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl", header=None)
    df = df.iloc[1:].copy()
    sub = df.iloc[:, [0, 1, 2]].copy()
    sub.columns = ["start_frame", "end_frame", "class_raw"]
    return sub


def _load_v3(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl", header=0)
    sub = df.iloc[:, [0, 1, 2]].copy()
    sub.columns = ["start_frame", "end_frame", "class_raw"]
    return sub


def _load_lower_start_end_class(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl", header=0)
    df = _strip_columns(df)
    sub = df[["start", "end", "class"]].rename(
        columns={"start": "start_frame", "end": "end_frame", "class": "class_raw"}
    )
    return sub


def load_annotation_events(path: Path) -> tuple[pd.DataFrame, str]:
    """Return normalized events and a short loader label for the summary table."""
    vid = _video_id(path)
    loaders: dict[str, tuple[Callable[[Path], pd.DataFrame], str]] = {
        "V1": (lambda p: _load_v1_v10_style(p, "Start_Frame", "Ending_Frame", "Class"), "v1_start_ending_class"),
        "V2": (lambda p: _load_no_header_first3(p), "v2_no_header_cols012"),
        "V3": (lambda p: _load_v3(p), "v3_first_three_cols"),
        "V4": (lambda p: _load_v1_v10_style(p, "Start", "End", "Class"), "v4_start_end_class"),
        "V5": (lambda p: _load_no_header_first3(p), "v5_no_header_cols012"),
        "V6": (lambda p: _load_lower_start_end_class(p), "v6_start_end_class"),
        "V7": (lambda p: _load_lower_start_end_class(p), "v7_start_end_class"),
        "V8": (lambda p: _load_lower_start_end_class(p), "v8_start_end_class"),
        "V9": (
            lambda p: _load_v1_v10_style(p, "Start Frame", "End Frame", "Class"),
            "v9_start_frame_end_frame_class",
        ),
        "V10": (
            lambda p: _load_v1_v10_style(p, "Start Frame", "End Frame", "Class"),
            "v10_start_frame_end_frame_class",
        ),
    }
    if vid not in loaders:
        raise KeyError(f"No loader registered for {vid} ({path})")
    fn, label = loaders[vid]
    return fn(path), label


def annotate_quality(events: pd.DataFrame) -> pd.DataFrame:
    """Add duration, flags, and bad_reasons."""
    df = events.copy()
    s = pd.to_numeric(df["start_frame"], errors="coerce")
    e = pd.to_numeric(df["end_frame"], errors="coerce")
    df["start_frame"] = s
    df["end_frame"] = e

    cls = df["class_raw"]
    cls_str = cls.map(lambda x: str(x).strip() if pd.notna(x) and str(x).strip() != "nan" else np.nan)

    missing_start = df["start_frame"].isna()
    missing_end = df["end_frame"].isna()
    missing_class = cls_str.isna() | (cls_str == "")

    both_num = df["start_frame"].notna() & df["end_frame"].notna()
    end_before_start = both_num & (df["end_frame"] < df["start_frame"])

    duration = np.where(
        both_num & ~end_before_start,
        df["end_frame"] - df["start_frame"] + 1,
        np.nan,
    )
    df["duration_inclusive"] = duration
    df["class_raw"] = cls_str

    warn_long = both_num & ~end_before_start & (df["duration_inclusive"] > DURATION_WARN_FRAMES)

    def build_reasons(i: int) -> str:
        parts: list[str] = []
        if missing_start.iloc[i]:
            parts.append("missing_start")
        if missing_end.iloc[i]:
            parts.append("missing_end")
        if missing_class.iloc[i]:
            parts.append("missing_class")
        if end_before_start.iloc[i]:
            parts.append("end_before_start")
        if warn_long.iloc[i]:
            parts.append(f"long_event_over_{DURATION_WARN_FRAMES}_frames")
        return ";".join(parts)

    idx = range(len(df))
    df["bad_reasons"] = [build_reasons(i) for i in idx]

    is_bad_core = missing_start | missing_end | missing_class | end_before_start
    df["is_bad"] = is_bad_core

    return df


def summarize_video(path: Path, events_q: pd.DataFrame, loader_kind: str) -> dict:
    vid = _video_id(path)
    ok = events_q[~events_q["is_bad"]].copy()
    durations = ok["duration_inclusive"].dropna()
    labels = sorted({str(x) for x in ok["class_raw"].dropna().unique()})
    return {
        "video_id": vid,
        "source_file": path.name,
        "loader_kind": loader_kind,
        "num_rows": len(events_q),
        "num_bad_rows": int(events_q["is_bad"].sum()),
        "num_long_events_warn": int(
            events_q["bad_reasons"].str.contains(f"long_event_over_{DURATION_WARN_FRAMES}", regex=False).sum()
        ),
        "unique_label_count": len(labels),
        "unique_labels": "|".join(labels),
        "min_duration_ok": float(durations.min()) if len(durations) else np.nan,
        "max_duration_ok": float(durations.max()) if len(durations) else np.nan,
        "mean_duration_ok": float(durations.mean()) if len(durations) else np.nan,
        "median_duration_ok": float(durations.median()) if len(durations) else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit annotation .xlsx files.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/Annotation_files"),
        help="Directory containing V*.xlsx files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/reports"),
        help="Directory for CSV reports (created if missing)",
    )
    parser.add_argument(
        "--no-events",
        action="store_true",
        help="Only write annotation_audit_summary.csv",
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("V*.xlsx"))
    if not files:
        raise SystemExit(f"No V*.xlsx files under {input_dir}")

    summary_rows: list[dict] = []
    event_parts: list[pd.DataFrame] = []

    for path in files:
        events, loader_kind = load_annotation_events(path)
        events = events.reset_index(drop=True)
        events_q = annotate_quality(events)
        events_q.insert(0, "video_id", _video_id(path))
        events_q.insert(1, "source_file", path.name)
        events_q.insert(2, "row_index", np.arange(len(events_q), dtype=int))
        summary_rows.append(summarize_video(path, events_q, loader_kind))
        if not args.no_events:
            event_parts.append(events_q)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "annotation_audit_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path} ({len(summary_df)} videos)")

    if not args.no_events:
        all_events = pd.concat(event_parts, ignore_index=True)
        events_path = output_dir / "annotation_audit_events.csv"
        all_events.to_csv(events_path, index=False)
        print(f"Wrote {events_path} ({len(all_events)} rows)")


if __name__ == "__main__":
    main()
