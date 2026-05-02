#!/usr/bin/env python3
"""API interface for VGG-16 model integration."""

from pathlib import Path
from typing import Dict, Any

from ..base.api import BaseModelAPI
from . import pipeline as vgg16_impl


class VGG16API(BaseModelAPI):
    """API for VGG-16 with transfer learning."""

    name = "vgg_16"

    def __init__(self):
        super().__init__()
        self.default_config = Path(__file__).parent / "hparams.yaml"
        self.defaults = {
            "seed": 42,
            "device": "auto",
            "splits": {"train": 0.90, "val": 0.05, "test": 0.05},
            "patience": 10,
            "lr_patience": 5,
            "use_class_weights": True,
            "class_names": "straight,hook,uppercut,none",
            "epochs": 30,
            "batch_size": 16,
            "image_size": 224,
            "num_workers": 2,
            "frames_per_video": 1,  # Not used, but required by base pipeline
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "output_dir": "runs/vgg_16",
            "run_name": "vgg_16_multiclass",
            "report_dir": "runs/vgg_16/eval"
        }

    def train(self, raw_args: list[str]) -> int:
        return self._invoke(mode="train", raw_args=raw_args, runner=vgg16_impl.train, default_config=self.default_config, defaults=self.defaults)

    def test(self, raw_args: list[str]) -> int:
        return self._invoke(mode="test", raw_args=raw_args, runner=vgg16_impl.test_only, default_config=self.default_config, defaults=self.defaults)

    def validate(self, raw_args: list[str]) -> int:
        return self._invoke(mode="validate", raw_args=raw_args, runner=vgg16_impl.validate_only, default_config=self.default_config, defaults=self.defaults)
