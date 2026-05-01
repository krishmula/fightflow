#!/usr/bin/env python3
"""API interface for ResNet-18 model integration."""

from pathlib import Path
from typing import Dict, Any

from ..base.api import BaseModelAPI
from . import pipeline as resnet18_impl


class ResNet18API(BaseModelAPI):
    """API for ResNet-18 with transfer learning."""

    name = "resnet18"

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
            "output_dir": "runs/resnet18",
            "run_name": "resnet18_multiclass",
            "report_dir": "runs/resnet18/eval"
        }

    def train(self, raw_args: list[str]) -> int:
        return self._invoke(mode="train", raw_args=raw_args, runner=resnet18_impl.train, default_config=self.default_config, defaults=self.defaults)

    def test(self, raw_args: list[str]) -> int:
        return self._invoke(mode="test", raw_args=raw_args, runner=resnet18_impl.test_only, default_config=self.default_config, defaults=self.defaults)

    def validate(self, raw_args: list[str]) -> int:
        return self._invoke(mode="validate", raw_args=raw_args, runner=resnet18_impl.validate_only, default_config=self.default_config, defaults=self.defaults)