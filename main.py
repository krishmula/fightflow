#!/usr/bin/env python3
"""Single gateway CLI for model APIs, using explicit model + task dispatch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.cnn_baseline.api import BaselineCNNAPI
from src.models.cnn_advanced.api import AdvancedCNNAPI
from src.models.base.api import BaseModelAPI


def _clean_forwarded_args(raw_args: list[str]) -> list[str]:
    # Support both styles: `gateway train -- ...` and `gateway train ...`
    if raw_args and raw_args[0] == "--":
        return raw_args[1:]
    return raw_args


def _validate_model_folder(model_name: str) -> None:
    model_dir = _ROOT / "src" / "models" / model_name
    if not model_dir.is_dir():
        raise SystemExit(f"Error: Model directory not found: {model_dir}")

    required_files = {
        "api.py": "The entry point interface that exposes model tasks.",
        "pipeline.py": "The core logic, data loading, and execution code.",
        "hparams.yaml": "The default configuration and hyperparameters.",
        "README.md": "The documentation explaining the model setup and parameters.",
        "__init__.py": "Required to make the folder a valid Python package."
    }

    missing = []
    for filename, desc in required_files.items():
        if not (model_dir / filename).is_file():
            missing.append(f"  - {filename}: {desc}")

    if missing:
        msg = f"\nError: Model '{model_name}' is missing required plug-and-play files:\n"
        msg += "\n".join(missing)
        raise SystemExit(msg)


def _ensure_output_folders(model_name: str) -> None:
    """Creates output directories inside the model folder if they don't exist."""
    model_dir = _ROOT / "src" / "models" / model_name
    for folder in ["runs", "eval", "inference"]:
        (model_dir / folder).mkdir(exist_ok=True)


def _get_model_api(model_name: str):
    if model_name == "cnn_baseline":
        return BaselineCNNAPI()
    if model_name == "cnn_advanced":
        return AdvancedCNNAPI()
    raise SystemExit(f"Unknown model '{model_name}'. Available models: ['cnn_baseline', 'cnn_advanced']")


def _available_tasks(api_obj) -> list[str]:
    return sorted(
        name
        for name in dir(api_obj)
        if not name.startswith("_") and callable(getattr(api_obj, name))
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Model gateway: choose a model and task, then forward task-specific args."
        )
    )
    parser.add_argument("--model", type=str, default=None, help="Model name, e.g. cnn_baseline")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task name for the selected model (e.g. train, test, validate, overlay)",
    )
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Args forwarded to the selected model task")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    task_name = args.task.strip().replace("-", "_")
    
    # Special bypass: prepare_data is model-agnostic, run it purely off BaseModelAPI.
    if task_name == "prepare_data" and not args.model:
        model_api = BaseModelAPI()
    else:
        if not args.model:
            raise SystemExit(f"Error: You must specify --model when running the '{args.task}' task.")
            
        _validate_model_folder(args.model)
        _ensure_output_folders(args.model)
        model_api = _get_model_api(args.model)
        
    if not hasattr(model_api, task_name):
        raise SystemExit(
            f"Task '{args.task}' is not available for model '{args.model}'. "
            f"Available tasks: {_available_tasks(model_api)}"
        )

    task_fn = getattr(model_api, task_name)
    rc = task_fn(_clean_forwarded_args(args.args))

    raise SystemExit(rc)


if __name__ == "__main__":
    main()
