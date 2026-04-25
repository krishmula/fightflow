#!/usr/bin/env python3
"""Baseline CNN API adapter used by the model gateway."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import yaml

from models.baseline_cnn import pipeline as baseline_impl
from models.baseline_cnn import inference as inference_impl


class BaselineCNNAPI:
    """API surface for baseline CNN model operations."""

    name = "baseline_cnn"

    def train(self, raw_args: list[str]) -> int:
        return self._invoke(mode="train", raw_args=raw_args, runner=baseline_impl.train)

    def test(self, raw_args: list[str]) -> int:
        return self._invoke(mode="test", raw_args=raw_args, runner=baseline_impl.test_only)

    def validate(self, raw_args: list[str]) -> int:
        return self._invoke(mode="validate", raw_args=raw_args, runner=baseline_impl.validate_only)

    def inference(self, raw_args: list[str]) -> int:
        return self._invoke(mode="inference", raw_args=raw_args, runner=inference_impl.main)

    @staticmethod
    def _clean_args(raw_args: list[str]) -> list[str]:
        # Allow both styles: `... train -- --epochs 10` and `... train --epochs 10`.
        if raw_args and raw_args[0] == "--":
            return raw_args[1:]
        return raw_args

    @classmethod
    def _extract_config_path(cls, raw_args: list[str]) -> tuple[str | None, list[str]]:
        config_path: str | None = None
        filtered: list[str] = []
        i = 0
        while i < len(raw_args):
            token = raw_args[i]
            if token == "--config":
                if i + 1 >= len(raw_args):
                    raise SystemExit("--config requires a YAML file path")
                config_path = raw_args[i + 1]
                i += 2
                continue
            if token.startswith("--config="):
                config_path = token.split("=", 1)[1].strip()
                i += 1
                continue
            filtered.append(token)
            i += 1
            
        if config_path is None:
            default_config = Path(__file__).parent / "hparams.yaml"
            if default_config.exists():
                config_path = str(default_config)
                
        return config_path, filtered

    @staticmethod
    def _load_yaml(path_value: str) -> dict:
        cfg_path = baseline_impl.resolve_project_path(path_value)
        if not cfg_path.is_file():
            raise SystemExit(f"Config file not found: {cfg_path}")

        with Path(cfg_path).open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}

        if not isinstance(payload, dict):
            raise SystemExit("Config YAML must contain a mapping at the top level")
        return payload

    @staticmethod
    def _select_mode_config(payload: dict, mode: str) -> dict:
        mode_keys = {"common", "train", "test", "validate", "inference"}
        if any(k in payload for k in mode_keys):
            out: dict = {}
            common = payload.get("common", {})
            if common is not None:
                if not isinstance(common, dict):
                    raise SystemExit("Config 'common' section must be a mapping")
                out.update(common)

            mode_cfg = payload.get(mode, {})
            if mode_cfg is not None:
                if not isinstance(mode_cfg, dict):
                    raise SystemExit(f"Config '{mode}' section must be a mapping")
                out.update(mode_cfg)
            return out

        return payload

    def _invoke(self, mode: str, raw_args: list[str], runner: Callable) -> int:
        clean_args = self._clean_args(raw_args)
        config_path, forwarded = self._extract_config_path(clean_args)
        
        config_dict = {}
        if config_path:
            payload = self._load_yaml(config_path)
            config_dict = self._select_mode_config(payload, mode)
            
        # Optional: override with forwarded args like `--epochs 10`
        i = 0
        while i < len(forwarded):
            if forwarded[i].startswith("--"):
                key = forwarded[i][2:].replace("-", "_")
                if i + 1 < len(forwarded) and not forwarded[i+1].startswith("--"):
                    val = forwarded[i+1]
                    if val.isdigit(): val = int(val)
                    else:
                        try: val = float(val)
                        except ValueError: pass
                    config_dict[key] = val
                    i += 2
                else:
                    config_dict[key] = True
                    i += 1
            else:
                i += 1

        # Ensure minimal defaults are set if missing from hparams
        defaults = {
            "seed": 42,
            "device": "auto",
            "val_ratio": 0.2,
            "patience": 6,
            "lr_patience": 2,
            "use_class_weights": False,
            "data_root": "data/classification/boxing4cls",
            "class_names": "jab,not_jab",
            "epochs": 20,
            "batch_size": 32,
            "image_size": 128,
            "num_workers": 2,
            "frames_per_video": 10,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "output_dir": "models/baseline_cnn/runs",
            "run_name": "baseline",
            "report_dir": "models/baseline_cnn/eval"
        }
        for k, v in defaults.items():
            if k not in config_dict:
                config_dict[k] = v

        from argparse import Namespace
        runner(Namespace(**config_dict))
        return 0

