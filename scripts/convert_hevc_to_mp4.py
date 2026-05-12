#!/usr/bin/env python3
"""
Convert iPhone HEVC (.mov) videos to H.264 MP4.

Usage:
    python scripts/convert_hevc_to_mp4.py <input> [<input> ...] [options]

Examples:
    python scripts/convert_hevc_to_mp4.py video.mov
    python scripts/convert_hevc_to_mp4.py *.mov --output-dir ./converted
    python scripts/convert_hevc_to_mp4.py ./raw_videos/ --recursive
"""

import argparse
import subprocess
import sys
from pathlib import Path

HEVC_EXTENSIONS = {".mov", ".hevc", ".mp4"}


def find_videos(paths: list[Path], recursive: bool) -> list[Path]:
    videos = []
    for path in paths:
        if path.is_dir():
            pattern = "**/*" if recursive else "*"
            for ext in HEVC_EXTENSIONS:
                videos.extend(path.glob(f"{pattern}{ext}"))
        elif path.suffix.lower() in HEVC_EXTENSIONS:
            videos.append(path)
        else:
            print(f"Skipping {path}: unsupported extension")
    return sorted(set(videos))


def convert(input_path: Path, output_dir: Path | None, overwrite: bool) -> bool:
    out_dir = output_dir or input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / (input_path.stem + ".mp4")

    if output_path == input_path:
        # Avoid clobbering the source if it's already .mp4
        output_path = out_dir / (input_path.stem + "_converted.mp4")

    if output_path.exists() and not overwrite:
        print(f"  Skipping (exists): {output_path}")
        return True

    cmd = [
        "ffmpeg",
        "-i", str(input_path),
        "-c:v", "libx264",       # Re-encode to H.264 for broad compatibility
        "-crf", "18",            # High quality (lower = better; 18 is visually lossless)
        "-preset", "fast",
        "-c:a", "aac",           # Re-encode audio to AAC
        "-b:a", "192k",
        "-movflags", "+faststart",  # Optimize for streaming
        "-y" if overwrite else "-n",
        str(output_path),
    ]

    print(f"  {input_path.name} → {output_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ERROR converting {input_path.name}:")
        print(result.stderr[-500:])  # tail of ffmpeg stderr
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert iPhone HEVC videos to H.264 MP4")
    parser.add_argument("inputs", nargs="+", type=Path, help="Files or directories to convert")
    parser.add_argument("--output-dir", "-o", type=Path, default=None,
                        help="Directory for converted files (default: same as source)")
    parser.add_argument("--recursive", "-r", action="store_true",
                        help="Search directories recursively")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output files")
    args = parser.parse_args()

    videos = find_videos(args.inputs, args.recursive)
    if not videos:
        print("No HEVC/MOV videos found.")
        sys.exit(1)

    print(f"Found {len(videos)} video(s) to convert.\n")
    ok, failed = 0, 0
    for v in videos:
        success = convert(v, args.output_dir, args.overwrite)
        if success:
            ok += 1
        else:
            failed += 1

    print(f"\nDone: {ok} converted, {failed} failed.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
