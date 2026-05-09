#!/usr/bin/env python3
"""Advanced CNN API adapter used by the model gateway."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from ..base.api import BaseModelAPI
from . import pipeline as advanced_impl


class AdvancedCNNAPI(BaseModelAPI):
    """API surface for advanced CNN model operations."""

    name = "advanced_cnn"

    def __init__(self):
        super().__init__()
        self.default_config = Path(__file__).parent / "hparams.yaml"

    def train(self, raw_args: list[str]) -> int:
        return self._invoke(mode="train", raw_args=raw_args, runner=advanced_impl.train, default_config=self.default_config)

    def test(self, raw_args: list[str]) -> int:
        return self._invoke(mode="test", raw_args=raw_args, runner=advanced_impl.test_only, default_config=self.default_config)

    def validate(self, raw_args: list[str]) -> int:
        return self._invoke(mode="validate", raw_args=raw_args, runner=advanced_impl.validate_only, default_config=self.default_config)
