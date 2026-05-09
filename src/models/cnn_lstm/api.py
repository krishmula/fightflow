#!/usr/bin/env python3
"""API adapter for CNN+LSTM model operations."""

from __future__ import annotations

from pathlib import Path

from ..base.api import BaseModelAPI
from . import pipeline as cnn_lstm_impl


class CNNLSTMAPI(BaseModelAPI):
    """API surface for CNN+LSTM operations."""

    name = "cnn_lstm"

    def __init__(self):
        super().__init__()
        self.default_config = Path(__file__).parent / "hparams.yaml"

    def train(self, raw_args: list[str]) -> int:
        return self._invoke(mode="train", raw_args=raw_args, runner=cnn_lstm_impl.train, default_config=self.default_config)

    def test(self, raw_args: list[str]) -> int:
        return self._invoke(mode="test", raw_args=raw_args, runner=cnn_lstm_impl.test_only, default_config=self.default_config)

    def validate(self, raw_args: list[str]) -> int:
        return self._invoke(mode="validate", raw_args=raw_args, runner=cnn_lstm_impl.validate_only, default_config=self.default_config)
