#!/usr/bin/env python3
"""Audit skeleton .npy files: shapes, value ranges, zero/missing frames, vs annotations.

Assumes typical layout (N, T, J, C) with C in {2, 3} for xy or xy+confidence.
Writes CSV summary, a short markdown report, and a few PNG visualizations.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Non-interactive backend before pyplot import (avoids GUI/sandbox crashes).
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# COCO-17 limb pairs for stick-figure plots (topology assumed; confirm with domain expert).
COCO17_EDGES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def _video_id(path: Path) -> str:
    m = re.match(r"^(V\d+)$", path.stem, re.I)
    if not m:
        raise ValueError(f"Expected filename like V7.npy, got {path.name}")
    return m.group(1).upper()


def _is_zero_frame(frame: np.ndarray, atol: float = 1e-9) -> bool:
    """True if all joint coords in this frame are ~0."""
    return bool(np.allclose(frame, 0.0, atol=atol))


def audit_array(arr: np.ndarray) -> dict:
    """Statistics for one loaded array (works with mmap; does not copy full array)."""
    flat = arr.ravel()
    pct_zero = 100.0 * float(np.mean(np.isclose(flat, 0.0)))
    return {
        "shape": str(arr.shape),
        "ndim": arr.ndim,
        "dtype": str(arr.dtype),
        "min_val": float(np.nanmin(flat)),
        "max_val": float(np.nanmax(flat)),
        "pct_values_near_zero": pct_zero,
    }


def per_event_missing_stats(arr: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Return mean zero-frame rate per event, fraction of events with >50% zero frames, per-event rates."""
    if arr.ndim != 4:
        return float("nan"), float("nan"), np.array([])

    n, t, j, c = arr.shape
    rates = np.zeros(n, dtype=np.float64)
    for e in range(n):
        ev = arr[e]
        z = sum(1 for f in range(t) if _is_zero_frame(ev[f]))
        rates[e] = z / t if t else 0.0
    mean_rate = float(np.mean(rates)) if n else float("nan")
    bad_events = float(np.mean(rates > 0.5)) if n else float("nan")
    return mean_rate, bad_events, rates


def infer_dimensions(ndim: int, shape: tuple[int, ...]) -> str:
    if ndim == 4:
        n, t, j, c = shape
        coord = "x,y" if c == 2 else ("x,y,conf" if c == 3 else f"{c} channels")
        return (
            f"dim0={n} (events), dim1={t} (frames per clip), "
            f"dim2={j} (joints), dim3={c} ({coord})"
        )
    return f"unexpected layout: {shape}"


def plot_skeleton_sequence(
    clip: np.ndarray,
    out_path: Path,
    title: str,
    max_frames: int = 8,
) -> None:
    """clip: (T, J, 2) or (T, J, 3) — uses xy only."""
    t = clip.shape[0]
    frames_idx = np.linspace(0, t - 1, num=min(max_frames, t), dtype=int)
    ncols = len(frames_idx)
    fig, axes = plt.subplots(1, ncols, figsize=(2.2 * ncols, 3), squeeze=False)
    xy = clip[..., :2]
    # Shared bounds for consistent scale
    xmin, xmax = float(xy[..., 0].min()), float(xy[..., 0].max())
    ymin, ymax = float(xy[..., 1].min()), float(xy[..., 1].max())
    pad = 0.02 * max(xmax - xmin, ymax - ymin, 1e-6)
    for ax, fi in zip(axes[0], frames_idx):
        pts = xy[fi]
        ax.scatter(pts[:, 0], pts[:, 1], s=12, c="tab:blue", zorder=2)
        for a, b in COCO17_EDGES:
            if a < pts.shape[0] and b < pts.shape[0]:
                ax.plot(
                    [pts[a, 0], pts[b, 0]],
                    [pts[a, 1], pts[b, 1]],
                    "k-",
                    linewidth=0.8,
                    alpha=0.6,
                )
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymax + pad, ymin - pad)  # image coords: y down
        ax.set_aspect("equal")
        ax.set_title(f"f={fi}")
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit skeleton V*.npy files.")
    parser.add_argument(
        "--skeleton-dir",
        type=Path,
        default=Path("data/raw/Skeleton_data"),
        help="Directory containing V*.npy",
    )
    parser.add_argument(
        "--annotation-summary",
        type=Path,
        default=Path("data/processed/reports/annotation_audit_summary.csv"),
        help="CSV with video_id and num_rows from Step 2",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/reports"),
        help="Reports and viz output directory",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip PNG visualizations",
    )
    args = parser.parse_args()

    sk_dir: Path = args.skeleton_dir
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = out_dir / "skeleton_viz"

    ann_rows: dict[str, int] = {}
    if args.annotation_summary.exists():
        adf = pd.read_csv(args.annotation_summary)
        for _, row in adf.iterrows():
            ann_rows[str(row["video_id"])] = int(row["num_rows"])

    files = sorted(sk_dir.glob("V*.npy"))
    if not files:
        raise SystemExit(f"No V*.npy files under {sk_dir}")

    summary_rows: list[dict] = []

    for path in files:
        vid = _video_id(path)
        arr = np.load(path, mmap_mode="r")
        base = audit_array(arr)
        mean_zero_frame_rate, frac_bad_events, _ = per_event_missing_stats(arr)

        num_events = int(arr.shape[0]) if arr.ndim >= 1 else 0
        ann_n = ann_rows.get(vid)
        match = (
            "yes"
            if ann_n is not None and num_events == ann_n
            else ("no" if ann_n is not None else "unknown")
        )

        row = {
            "video_id": vid,
            "source_file": path.name,
            **base,
            "num_events_dim0": num_events,
            "annotation_num_rows": ann_n if ann_n is not None else "",
            "event_count_matches_annotation": match,
            "dimension_interpretation": infer_dimensions(arr.ndim, arr.shape),
            "mean_zero_frame_rate_per_event": mean_zero_frame_rate,
            "frac_events_over_half_zero_frames": frac_bad_events,
        }
        if arr.ndim == 4:
            row["frames_per_event"] = int(arr.shape[1])
            row["num_joints"] = int(arr.shape[2])
            row["coord_channels"] = int(arr.shape[3])
        else:
            row["frames_per_event"] = ""
            row["num_joints"] = ""
            row["coord_channels"] = ""

        summary_rows.append(row)

        if not args.no_viz and arr.ndim == 4 and arr.shape[0] > 0:
            clip0 = np.asarray(arr[0], dtype=np.float64)
            # xy only for 3-channel (e.g. x, y, confidence)
            if clip0.shape[-1] == 3:
                clip0 = clip0[..., :2].copy()
            plot_skeleton_sequence(
                clip0,
                viz_dir / f"{vid}_first_event_frames.png",
                title=f"{vid} — first annotated event (skeleton sequence)",
            )

    summary_df = pd.DataFrame(summary_rows)
    csv_path = out_dir / "skeleton_audit_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    # Markdown report (Step 3 deliverable)
    report_path = out_dir / "skeleton_audit_report.md"
    lines = [
        "# Skeleton audit (Step 3)",
        "",
        "Generated by `scripts/audit_skeletons.py`.",
        "",
        "## Per-file summary",
        "",
        "See `skeleton_audit_summary.csv` for full columns.",
        "",
    ]
    for _, r in summary_df.iterrows():
        lines.append(f"### {r['video_id']}")
        lines.append("")
        lines.append(f"- **Shape / dtype**: `{r['shape']}` / {r['dtype']}")
        lines.append(f"- **Value range**: [{r['min_val']:.6g}, {r['max_val']:.6g}]")
        lines.append(f"- **% of values ≈0**: {r['pct_values_near_zero']:.4f}")
        lines.append(f"- **Interpretation**: {r['dimension_interpretation']}")
        lines.append(
            f"- **Annotation row count match**: {r['event_count_matches_annotation']} "
            f"(skeleton events={r['num_events_dim0']}, annotation rows={r['annotation_num_rows']!s})"
        )
        if pd.notna(r.get("mean_zero_frame_rate_per_event")):
            lines.append(
                f"- **Missing frames (mean all-zero frame rate / event)**: "
                f"{r['mean_zero_frame_rate_per_event']:.4f}"
            )
            lines.append(
                f"- **Fraction of events with >50% all-zero frames**: "
                f"{r['frac_events_over_half_zero_frames']:.4f}"
            )
        lines.append("")

    anomalies = summary_df[summary_df["event_count_matches_annotation"] != "yes"]
    if len(anomalies):
        lines.extend(["## Count or format anomalies", ""])
        for _, r in anomalies.iterrows():
            lines.append(f"### {r['video_id']}")
            lines.append("")
            if r["video_id"] == "V6":
                lines.extend(
                    [
                        "- **Skeleton rows vs annotations**: "
                        f"{r['num_events_dim0']} skeleton indices vs {r['annotation_num_rows']} "
                        "annotation rows — **not** a 1:1 event clip layout.",
                        "- **Shape** `(N, 1, 17, 3)`: one time step per row; last channel may be "
                        "confidence or an extra coordinate. **Large value range** (see `max_val` "
                        "in CSV) vs ~[0,1] on other videos suggests **pixel-scale or mixed** coords.",
                        "- **Next step**: resolve mapping in alignment (Step 7): e.g. frame-indexed "
                        "pose dump vs per-event windows.",
                        "",
                    ]
                )
            else:
                lines.append(
                    f"- **Match status**: {r['event_count_matches_annotation']}; "
                    f"skeleton events={r['num_events_dim0']}, annotation rows={r['annotation_num_rows']!s}."
                )
                lines.append("")

    lines.extend(
        [
            "## Coordinate scale",
            "",
            "If values fall roughly in [0, 1], they are likely normalized bounding-box or "
            "image coordinates. Compare `min_val` / `max_val` in the CSV.",
            "",
            "**Sparse zeros**: a high share of coordinates near zero can mean occluded or unused "
            "joints (not only “missing frame”). Pair with “all-zero frame” rates above.",
            "",
            "## Visualizations",
            "",
            "PNG sequences (first event per video) are under `skeleton_viz/`. "
            "Limbs use **COCO-17** connectivity as a drawing aid; joint order may differ — "
            "validate against the dataset paper or authors if poses look wrong.",
            "",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report_path}")
    if not args.no_viz:
        print(f"Wrote figures under {viz_dir}")


if __name__ == "__main__":
    main()
