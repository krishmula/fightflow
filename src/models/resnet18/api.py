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

    def train(self, raw_args: list[str]) -> int:
        return self._invoke(mode="train", raw_args=raw_args, runner=resnet18_impl.train, default_config=self.default_config)

    def test(self, raw_args: list[str]) -> int:
        return self._invoke(mode="test", raw_args=raw_args, runner=resnet18_impl.test_only, default_config=self.default_config)

    def validate(self, raw_args: list[str]) -> int:
        return self._invoke(mode="validate", raw_args=raw_args, runner=resnet18_impl.validate_only, default_config=self.default_config)