from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def is_improvement(value: float, best: float | None, mode: str) -> bool:
    if best is None:
        return True
    if mode == "min":
        return value < best
    if mode == "max":
        return value > best
    raise ValueError(f"Unknown checkpoint mode: {mode}")


def initial_best(mode: str) -> float:
    if mode == "min":
        return float("inf")
    if mode == "max":
        return -float("inf")
    raise ValueError(f"Unknown mode: {mode}")


def capture_rng_state() -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng_state(rng_state: dict[str, Any] | None) -> None:
    if not rng_state:
        return
    torch_state = _to_cpu_byte_tensor(rng_state.get("torch"))
    if torch_state is not None:
        torch.set_rng_state(torch_state)
    else:
        print("Warning: skipped restoring torch RNG state because checkpoint format was incompatible.")

    cuda_state = rng_state.get("cuda")
    if torch.cuda.is_available() and cuda_state is not None:
        converted_cuda_state = [_to_cpu_byte_tensor(state) for state in cuda_state]
        if all(state is not None for state in converted_cuda_state):
            torch.cuda.set_rng_state_all(converted_cuda_state)
        else:
            print("Warning: skipped restoring CUDA RNG state because checkpoint format was incompatible.")

    if rng_state.get("numpy") is not None:
        np.random.set_state(rng_state["numpy"])
    if rng_state.get("python") is not None:
        random.setstate(rng_state["python"])


def _to_cpu_byte_tensor(value: Any) -> torch.ByteTensor | None:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(dtype=torch.uint8)
    if isinstance(value, (bytes, bytearray)):
        return torch.tensor(list(value), dtype=torch.uint8)
    if isinstance(value, (list, tuple)):
        try:
            return torch.tensor(value, dtype=torch.uint8)
        except (TypeError, ValueError):
            return None
    return None


def make_checkpoint(
    epoch: int,
    global_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler,
    amp_scaler,
    best_metric: float,
    best_epoch: int,
    epochs_without_improvement: int,
    config: dict,
    wandb_run_id: str | None,
    checkpoint_config: dict,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "epochs_without_improvement": epochs_without_improvement,
        "config": config,
        "wandb_run_id": wandb_run_id,
    }
    if checkpoint_config.get("save_optimizer_state", True) and optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if checkpoint_config.get("save_scheduler_state", True) and scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if checkpoint_config.get("save_amp_scaler_state", True) and amp_scaler is not None:
        state["amp_scaler_state_dict"] = amp_scaler.state_dict()
    if checkpoint_config.get("save_rng_state", True):
        state["rng_state"] = capture_rng_state()
    return state


def save_checkpoint(path: str | Path, state: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def resolve_resume_checkpoint(
    checkpoint_dir: str | Path,
    checkpoint_path: str | None,
    mode: str,
) -> Path | None:
    if checkpoint_path and checkpoint_path != "auto":
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
        return path

    checkpoint_dir = Path(checkpoint_dir)
    for name in ("last.pt", "best.pt"):
        path = checkpoint_dir / name
        if path.exists():
            return path
    if checkpoint_path == "auto":
        print(f"No checkpoint found in {checkpoint_dir}; starting fresh.")
    return None


def load_checkpoint(path: str | Path, map_location: torch.device | str) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def restore_checkpoint_state(
    checkpoint: dict[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler,
    amp_scaler,
    resume_mode: str,
    reset_optimizer: bool = False,
    reset_scheduler: bool = False,
) -> dict[str, int | float | None]:
    """Restore a checkpoint according to the selected resume mode.

    `full` is for crash recovery, `weights_only` is for architecture-compatible
    fine-tuning with new optimizer settings, and `model_optimizer` restores the
    optimizer while allowing scheduler reset.
    """
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as exc:
        raise RuntimeError(
            "Could not load model weights. This usually means architecture dimensions changed "
            "(hidden_dim, num_layers, embedding dimensions, or MLP width). Use matching model "
            "settings or start a fresh run."
        ) from exc

    if resume_mode == "weights_only":
        return {
            "start_epoch": 0,
            "global_step": 0,
            "best_metric": None,
            "best_epoch": -1,
            "epochs_without_improvement": 0,
        }

    if resume_mode in {"full", "model_optimizer", "auto"}:
        if (
            optimizer is not None
            and not reset_optimizer
            and "optimizer_state_dict" in checkpoint
            and resume_mode in {"full", "model_optimizer", "auto"}
        ):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if (
            scheduler is not None
            and not reset_scheduler
            and "scheduler_state_dict" in checkpoint
            and resume_mode in {"full", "auto"}
        ):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if amp_scaler is not None and "amp_scaler_state_dict" in checkpoint and resume_mode in {"full", "auto"}:
            amp_scaler.load_state_dict(checkpoint["amp_scaler_state_dict"])
        if resume_mode in {"full", "auto"}:
            restore_rng_state(checkpoint.get("rng_state"))
        return {
            "start_epoch": int(checkpoint.get("epoch", -1)) + 1,
            "global_step": int(checkpoint.get("global_step", 0)),
            "best_metric": checkpoint.get("best_metric"),
            "best_epoch": int(checkpoint.get("best_epoch", -1)),
            "epochs_without_improvement": int(checkpoint.get("epochs_without_improvement", 0)),
        }

    raise ValueError(f"Unknown resume_mode: {resume_mode}")
