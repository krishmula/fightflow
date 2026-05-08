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
        self.defaults = {
            "seed": 42,
            "device": "auto",
            "splits": {"train": 0.70, "val": 0.15, "test": 0.15},
            "patience": 6,
            "lr_patience": 2,
            "use_class_weights": True,
            "class_names": "straight,hook,uppercut,none",
            "epochs": 20,
            "batch_size": 8,
            "image_size": 224,
            "num_workers": 2,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "output_dir": "runs/cnn_lstm",
            "run_name": "cnn_lstm",
            "report_dir": "runs/cnn_lstm/eval",
            "clip_length": 16,
            "cnn_backbone": "resnet18",
            "cnn_init": "latest",
            "freeze_backbone": True,
            "lstm_hidden": 256,
            "lstm_layers": 1,
            "lstm_dropout": 0.0,
            "lstm_bidirectional": False,
            "lstm_output": "last",
            "feature_dim": 512,
            "augment": True,
        }

    def train(self, raw_args: list[str]) -> int:
        return self._invoke(mode="train", raw_args=raw_args, runner=cnn_lstm_impl.train, default_config=self.default_config, defaults=self.defaults)

    def test(self, raw_args: list[str]) -> int:
        return self._invoke(mode="test", raw_args=raw_args, runner=cnn_lstm_impl.test_only, default_config=self.default_config, defaults=self.defaults)

    def validate(self, raw_args: list[str]) -> int:
        return self._invoke(mode="validate", raw_args=raw_args, runner=cnn_lstm_impl.validate_only, default_config=self.default_config, defaults=self.defaults)

    def prepare_data(self, raw_args: list[str]) -> int:
        args = self._parse_args(raw_args, mode="prepare_data", default_config=self.default_config)
        super().prepare_data(args)
        return 0
