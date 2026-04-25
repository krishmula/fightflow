#!/usr/bin/env python3
"""Run frame-wise CNN classification on a video and render labels onto output video."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch
from PIL import Image
from tqdm import tqdm

try:
    from torchvision.models import get_model, get_model_weights
except ImportError:  # pragma: no cover
    get_model = None
    get_model_weights = None



def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_pretrained_model(model_name: str, weight_name: str, device: torch.device):
    if get_model is None or get_model_weights is None:
        raise RuntimeError(
            "Installed torchvision does not support get_model/get_model_weights. "
            "Please upgrade torchvision."
        )

    try:
        weights_enum = get_model_weights(model_name)
    except Exception as exc:
        raise ValueError(f"Unknown torchvision model: {model_name}") from exc

    try:
        weights = weights_enum.DEFAULT if weight_name.upper() == "DEFAULT" else getattr(weights_enum, weight_name)
    except AttributeError as exc:
        choices = [k for k in dir(weights_enum) if k.isupper()]
        raise ValueError(f"Unknown weight '{weight_name}'. Available: {choices}") from exc

    model = get_model(model_name, weights=weights)
    model.eval().to(device)
    preprocess = weights.transforms()
    labels = weights.meta.get("categories", [])
    if not labels:
        raise RuntimeError("Loaded weights do not include class category names.")
    return model, preprocess, labels


def predict_frame(
    frame_bgr,
    model,
    preprocess,
    labels: list[str],
    device: torch.device,
) -> tuple[str, float]:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    x = preprocess(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits[0], dim=0)
        score, idx = torch.max(probs, dim=0)

    return labels[int(idx)], float(score)


def draw_label(frame_bgr, label: str, score: float):
    h, w = frame_bgr.shape[:2]
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame_bgr, 0.5, 0, frame_bgr)

    text = f"class: {label}  conf: {score:.1%}"
    cv2.putText(
        frame_bgr,
        text,
        (12, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame_bgr


def main(args) -> None:
    input_video = Path(args.input_video)
    output_video = Path(args.output_video)

    if not input_video.is_file():
        raise SystemExit(f"Input video not found: {input_video}")
    if len(args.codec) != 4:
        raise SystemExit("--codec must be exactly 4 characters, e.g. mp4v")
    if args.frame_stride < 1:
        raise SystemExit("--frame-stride must be >= 1")

    device = resolve_device(args.device)
    model, preprocess, labels = load_pretrained_model(args.model_name, args.weights, device)

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open input video: {input_video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    in_fps = float(cap.get(cv2.CAP_PROP_FPS))
    fps = args.fps if args.fps > 0 else (in_fps if in_fps > 0 else 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_video.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"Could not open output video writer: {output_video}")

    print(f"Using device: {device}")
    print(f"Model: {args.model_name} ({args.weights})")
    print(f"Input: {input_video}")
    print(f"Output: {output_video}")

    idx = 0
    cached_label = ""
    cached_score = 0.0
    total = frame_count if frame_count > 0 else None

    with tqdm(total=total, desc="Processing frames", unit="frame") as pbar:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if idx % args.frame_stride == 0 or not cached_label:
                cached_label, cached_score = predict_frame(frame, model, preprocess, labels, device)

            out_frame = draw_label(frame, cached_label, cached_score)
            writer.write(out_frame)

            idx += 1
            pbar.update(1)

    cap.release()
    writer.release()
    print(f"Done. Wrote {idx} frames to {args.output_video}")


if __name__ == "__main__":
    main()
