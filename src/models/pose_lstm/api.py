#!/usr/bin/env python3
"""API adapter for Pose LSTM model operations."""

from __future__ import annotations
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from ..base.api import BaseModelAPI
from ..base.utils import resolve_project_path
from . import pipeline as pose_lstm_impl

class PoseLSTMAPI(BaseModelAPI):
    """API surface for Pose LSTM operations."""

    name = "pose_lstm"

    def __init__(self):
        super().__init__()
        self.default_config = Path(__file__).parent / "hparams.yaml"

    def train(self, raw_args: list[str]) -> int:
        return self._invoke(mode="train", raw_args=raw_args, runner=pose_lstm_impl.train, default_config=self.default_config)

    def test(self, raw_args: list[str]) -> int:
        return self._invoke(mode="test", raw_args=raw_args, runner=pose_lstm_impl.test_only, default_config=self.default_config)

    def validate(self, raw_args: list[str]) -> int:
        return self._invoke(mode="validate", raw_args=raw_args, runner=pose_lstm_impl.validate_only, default_config=self.default_config)

    def overlay(self, raw_args: list[str]) -> int:
        """Run inference on a video and save a visual overlay."""
        return self._invoke(mode="overlay", raw_args=raw_args, runner=pose_lstm_impl.overlay_video, default_config=self.default_config)
