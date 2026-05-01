#!/usr/bin/env python3
"""Baseline CNN API adapter used by the model gateway."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from ..base.api import BaseModelAPI
from . import pipeline as baseline_impl


class BaselineCNNAPI(BaseModelAPI):
    """API surface for baseline CNN model operations."""

    name = "cnn_baseline"

    def __init__(self):
        super().__init__()
        self.default_config = Path(__file__).parent / "hparams.yaml"
        self.defaults = {
            "seed": 42,
            "device": "auto",
            "splits": {"train": 0.70, "val": 0.15, "test": 0.15},
            "patience": 6,
            "lr_patience": 2,
            "use_class_weights": True,
            "class_names": ["straight", "hook", "uppercut", "none"],
            "epochs": 20,
            "batch_size": 32,
            "image_size": 128,
            "num_workers": 2,
            "frames_per_video": 10,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "output_dir": "runs/cnn_baseline",
            "run_name": "baseline_multiclass",
            "report_dir": "runs/cnn_baseline/eval"
        }

    def train(self, raw_args: list[str]) -> int:
        return self._invoke(mode="train", raw_args=raw_args, runner=baseline_impl.train, default_config=self.default_config, defaults=self.defaults)

    def test(self, raw_args: list[str]) -> int:
        return self._invoke(mode="test", raw_args=raw_args, runner=baseline_impl.test_only, default_config=self.default_config, defaults=self.defaults)

    def prepare_data(self, raw_args: list[str]) -> int:
        args = self._parse_args(raw_args, mode="prepare_data", default_config=self.default_config)
        super().prepare_data(args)
        return 0
