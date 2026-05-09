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

    def train(self, raw_args: list[str]) -> int:
        return self._invoke(mode="train", raw_args=raw_args, runner=vgg16_impl.train, default_config=self.default_config)

    def test(self, raw_args: list[str]) -> int:
        return self._invoke(mode="test", raw_args=raw_args, runner=vgg16_impl.test_only, default_config=self.default_config)

    def validate(self, raw_args: list[str]) -> int:
        return self._invoke(mode="validate", raw_args=raw_args, runner=vgg16_impl.validate_only, default_config=self.default_config)
