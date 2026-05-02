from __future__ import annotations

import copy
from pathlib import Path
from typing import Any


def require_yaml():
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required. Install it with: pip install PyYAML") from exc
    return yaml


def load_config(path: str | Path) -> dict[str, Any]:
    yaml = require_yaml()
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_by_path(config: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = config
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value


def apply_cli_overrides(config: dict[str, Any], args) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    overrides = {
        ("paths", "data_dir"): args.data_dir,
        ("paths", "checkpoint_dir"): args.checkpoint_dir,
        ("paths", "model_dir"): args.model_dir,
        ("paths", "log_dir"): args.log_dir,
        ("training", "batch_size"): args.batch_size,
        ("training", "learning_rate"): args.learning_rate,
        ("training", "max_epochs"): args.max_epochs,
        ("wandb", "enabled"): args.wandb_enabled,
    }
    for path, value in overrides.items():
        if value is not None:
            set_by_path(updated, path, value)

    if args.resume is not None:
        updated["resume"]["checkpoint_path"] = "auto" if args.resume == "auto" else args.resume
        updated["resume"]["mode"] = args.resume_mode or "auto"
    elif args.resume_mode is not None:
        updated["resume"]["mode"] = args.resume_mode
    return updated


def flatten_config(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in config.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_config(value, name))
        else:
            flat[name] = value
    return flat
