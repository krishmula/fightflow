#!/usr/bin/env python3
"""Step 4: read YouTube URLs from Meta_data.ods and download with yt-dlp.

Writes a CSV report (fps, duration, frames, status). Downloaded .mp4 files stay
under data/downloaded-videos/ by default (gitignored — do not commit videos).
By default, downloads are constrained to at most 1080p and 30 FPS.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

try:
    import cv2
except ImportError:
    cv2 = None


def _norm_video_id(s: str) -> str:
    t = str(s).strip().upper()
    m = re.match(r"^(V\d+)$", t)
    if not m:
        raise ValueError(f"Unexpected video id: {s!r}")
    return m.group(1)


def load_urls_from_ods(path: Path) -> list[tuple[str, str]]:
    df = pd.read_excel(path, engine="odf", header=None)
    rows: list[tuple[str, str]] = []
    for i in range(len(df)):
        vid_cell = df.iloc[i, 0]
        link_cell = df.iloc[i, 1] if df.shape[1] > 1 else None
        if str(vid_cell).strip().lower() == "video":
            continue
        if pd.isna(link_cell) or str(link_cell).strip() == "":
            continue
        vid = _norm_video_id(vid_cell)
        url = str(link_cell).strip()
        if not url.startswith("http"):
            continue
        rows.append((vid, url))

    def _sort_key(item: tuple[str, str]) -> int:
        m = re.search(r"(\d+)$", item[0])
        return int(m.group(1)) if m else 0

    rows.sort(key=_sort_key)
    return rows


def probe_ffprobe(path: Path) -> dict | None:
    try:
        r = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate,nb_frames",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError:
        return None
    streams = data.get("streams") or []
    fmt = data.get("format") or {}
    if not streams:
        return None
    st = streams[0]
    w = st.get("width")
    h = st.get("height")
    fps_s = st.get("avg_frame_rate") or ""
    nb = st.get("nb_frames")
    dur = fmt.get("duration")

    fps: float | None = None
    if isinstance(fps_s, str) and "/" in fps_s:
        a, b = fps_s.split("/", 1)
        try:
            fps = float(a) / float(b) if float(b) != 0 else None
        except ValueError:
            fps = None
    elif isinstance(fps_s, str):
        try:
            fps = float(fps_s)
        except ValueError:
            fps = None

    duration_sec: float | None = None
    if dur is not None:
        try:
            duration_sec = float(dur)
        except ValueError:
            duration_sec = None

    num_frames: int | None = None
    if nb not in (None, "N/A"):
        try:
            num_frames = int(nb)
        except ValueError:
            num_frames = None
    if num_frames is None and duration_sec is not None and fps is not None:
        num_frames = int(round(duration_sec * fps))

    return {
        "width": w,
        "height": h,
        "fps": fps,
        "duration_sec": duration_sec,
        "num_frames": num_frames,
        "probe": "ffprobe",
    }


def probe_opencv(path: Path) -> dict | None:
    if cv2 is None:
        return None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    fps_f = float(fps) if fps and fps > 0 else None
    n_i = int(n) if n and n > 0 else None
    duration_sec = None
    if fps_f and n_i:
        duration_sec = n_i / fps_f
    return {
        "width": w,
        "height": h,
        "fps": fps_f,
        "duration_sec": duration_sec,
        "num_frames": n_i,
        "probe": "opencv",
    }


def probe_video(path: Path) -> dict:
    if not path.is_file():
        return {
            "width": "",
            "height": "",
            "fps": "",
            "duration_sec": "",
            "num_frames": "",
            "probe": "",
        }
    for fn in (probe_ffprobe, probe_opencv):
        out = fn(path)
        if out is not None and out.get("fps") is not None:
            return out
    out = probe_opencv(path) or probe_ffprobe(path)
    if out is not None:
        return out
    return {
        "width": "",
        "height": "",
        "fps": "",
        "duration_sec": "",
        "num_frames": "",
        "probe": "failed",
    }


def build_ytdlp_command(
    url: str,
    out_path: Path,
    *,
    use_android_client: bool,
    max_height: int,
    max_fps: int,
) -> list[str]:
    """Build a yt-dlp command for MP4 output within the requested video bounds."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    template = out_path.with_suffix(".%(ext)s")
    # Use same interpreter as this script so the venv's yt-dlp is used (bare `yt-dlp`
    # is often missing from PATH when launching Python from an IDE or GUI).
    cmd: list[str] = [
        sys.executable,
        "-m",
        "yt_dlp",
    ]
    # The default client usually exposes higher-quality formats than android does.
    # We optionally fall back to android in case the default client hits YouTube 403s.
    if use_android_client:
        cmd.extend(["--extractor-args", "youtube:player_client=android"])
    cmd.extend(
        [
            "-f",
            (
                f"bv*[ext=mp4][height<={max_height}][fps<={max_fps}]+ba[ext=m4a]/"
                f"bv*[height<={max_height}][fps<={max_fps}]+ba/"
                f"b[ext=mp4][height<={max_height}][fps<={max_fps}]/"
                f"b[height<={max_height}][fps<={max_fps}]"
            ),
            "--merge-output-format",
            "mp4",
            "-o",
            str(template),
            "--no-playlist",
            "--newline",
            url,
        ]
    )
    return cmd


def run_ytdlp(
    url: str,
    out_path: Path,
    *,
    allow_android_fallback: bool,
    max_height: int,
    max_fps: int,
) -> tuple[int, str]:
    """Download merged MP4. Returns (returncode, log tail for failures)."""
    attempts = [False]
    if allow_android_fallback:
        attempts.append(True)
    logs: list[str] = []
    for use_android_client in attempts:
        cmd = build_ytdlp_command(
            url,
            out_path,
            use_android_client=use_android_client,
            max_height=max_height,
            max_fps=max_fps,
        )
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except FileNotFoundError:
            return 127, "python interpreter not found"
        client_name = "android" if use_android_client else "default"
        log = ((r.stderr or "") + "\n" + (r.stdout or ""))[-6000:]
        if r.returncode == 0:
            return 0, log
        logs.append(f"[{client_name} client]\n{log}")

    return r.returncode, "\n\n".join(logs)[-6000:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download source videos (Step 4).")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/raw/RGB_videos/Meta_data.ods"),
        help="Path to Meta_data.ods",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/downloaded-videos"),
        help="Directory for V*.mp4 (gitignored)",
    )
    parser.add_argument(
        "--no-android-client",
        action="store_true",
        help="Do not fall back to youtube:player_client=android if the default client fails",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/processed/reports/video_downloads.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print video_id and URL only; do not download",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip download if target .mp4 already exists",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=1080,
        help="Maximum downloaded video height in pixels",
    )
    parser.add_argument(
        "--max-fps",
        type=int,
        default=30,
        help="Maximum downloaded video frame rate",
    )
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated video ids to process (e.g. V1,V2)",
    )
    args = parser.parse_args()

    only: set[str] | None = None
    if args.only.strip():
        only = {_norm_video_id(x.strip()) for x in args.only.split(",") if x.strip()}

    meta: Path = args.metadata
    if not meta.is_file():
        raise SystemExit(f"Metadata not found: {meta}")

    pairs = load_urls_from_ods(meta)
    if only is not None:
        pairs = [(v, u) for v, u in pairs if v in only]
    if not pairs:
        raise SystemExit("No (video_id, url) rows parsed from metadata.")

    out_dir: Path = args.output_dir
    report_path: Path = args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    if args.dry_run:
        for vid, url in pairs:
            print(f"{vid}\t{url}")
        print(f"\nDry run: {len(pairs)} videos; would write to {out_dir}/")
        return

    for vid, url in pairs:
        target = out_dir / f"{vid}.mp4"
        status = "skipped_existing"
        if args.skip_existing and target.is_file():
            pass
        else:
            code, err = run_ytdlp(
                url,
                target,
                allow_android_fallback=not args.no_android_client,
                max_height=args.max_height,
                max_fps=args.max_fps,
            )
            if code != 0:
                status = f"error_code_{code}"
                sys.stderr.write(f"{vid}: yt-dlp failed ({code})\n{err}\n")
            elif not target.is_file():
                status = "error_missing_output"
            else:
                status = "ok"

        meta_p = probe_video(target)
        rows.append(
            {
                "video_id": vid,
                "source_url": url,
                "file_path": str(target.resolve()) if target.is_file() else "",
                "width": meta_p.get("width", ""),
                "height": meta_p.get("height", ""),
                "fps": meta_p.get("fps", ""),
                "duration_sec": meta_p.get("duration_sec", ""),
                "num_frames": meta_p.get("num_frames", ""),
                "probe": meta_p.get("probe", ""),
                "download_status": status,
            }
        )

    pd.DataFrame(rows).to_csv(report_path, index=False)
    print(f"Wrote {report_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
