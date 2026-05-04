from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.data.lstm_baseline.lstm_dataset import FastLSTMBatchBuilder
from src.models.baseline_lstm import BaselineLSTM
from src.models.tts_lstm import TTSLSTM
from src.training.lstm_training.checkpointing import (
    initial_best,
    is_improvement,
    load_checkpoint,
    make_checkpoint,
    resolve_resume_checkpoint,
    restore_checkpoint_state,
    save_checkpoint,
)
from src.training.lstm_training.config import apply_cli_overrides, flatten_config, load_config
from src.training.lstm_training.metrics import RawCountMetricAccumulator, inverse_transform_targets, load_target_scalers
from src.training.lstm_training.utils import (
    bool_from_string,
    detach_metric_dict,
    ensure_dirs,
    get_current_lr,
    load_json,
    select_device,
    set_seed,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Seoul bike LSTM.")
    parser.add_argument("--config", type=Path, default=Path("configs/models/lstm/lstm_v1.yaml"))
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--resume_mode", choices=["auto", "full", "weights_only", "model_optimizer"], default=None)
    parser.add_argument("--wandb_enabled", type=bool_from_string, default=None)
    parser.add_argument("--smoke_test", type=bool_from_string, default=True)
    parser.add_argument("--smoke_batch_size", type=int, default=512)
    return parser.parse_args()


def autocast_context(use_amp: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", enabled=use_amp)
    return torch.cuda.amp.autocast(enabled=use_amp)


def make_grad_scaler(use_amp: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=use_amp)
        except TypeError:
            return torch.amp.GradScaler(enabled=use_amp)
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def load_feature_metadata(data_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    return (
        load_json(data_dir / "dataset_summary.json"),
        load_json(data_dir / "feature_config.json"),
        load_json(data_dir / "district_vocab.json"),
    )


def normalize_architecture_name(value: str) -> str:
    return value.replace("-", "_").lower()


def build_model(config: dict, data_dir: Path, device: torch.device) -> torch.nn.Module:
    summary, feature_config, district_vocab = load_feature_metadata(data_dir)
    model_config = config["model"]
    architecture = normalize_architecture_name(str(model_config.get("architecture", "baseline_lstm")))
    common_kwargs = {
        "input_dim": len(feature_config["dynamic_feature_columns"]),
        "static_numeric_dim": len(feature_config["static_numeric_columns"]),
        "output_dim": len(feature_config["target_columns"]),
        "num_stations": int(summary["S"]),
        "num_districts": len(district_vocab),
    }
    if architecture == "baselinelstm":
        architecture = "baseline_lstm"

    if architecture == "baseline_lstm":
        model = BaselineLSTM(
            **common_kwargs,
            hidden_dim=int(model_config["hidden_dim"]),
            num_layers=int(model_config["num_layers"]),
            station_embedding_dim=int(model_config["station_embedding_dim"]),
            district_embedding_dim=int(model_config["district_embedding_dim"]),
            mlp_hidden_dim=int(model_config["mlp_hidden_dim"]),
            dropout=float(model_config["dropout"]),
        )
    elif architecture == "tts_lstm":
        model = TTSLSTM(
            **common_kwargs,
            window_offsets=feature_config["window_offsets"],
            recent_offsets=model_config["recent_offsets"],
            daily_offsets=model_config["daily_offsets"],
            weekly_offsets=model_config["weekly_offsets"],
            recent_hidden_dim=int(model_config["recent_hidden_dim"]),
            daily_hidden_dim=int(model_config["daily_hidden_dim"]),
            weekly_hidden_dim=int(model_config["weekly_hidden_dim"]),
            recent_num_layers=int(model_config["recent_num_layers"]),
            daily_num_layers=int(model_config["daily_num_layers"]),
            weekly_num_layers=int(model_config["weekly_num_layers"]),
            station_embedding_dim=int(model_config["station_embedding_dim"]),
            district_embedding_dim=int(model_config["district_embedding_dim"]),
            mlp_hidden_dims=model_config["mlp_hidden_dims"],
            dropout=float(model_config["dropout"]),
        )
    else:
        raise ValueError(f"Unsupported model architecture: {model_config.get('architecture')}")
    return model.to(device)


def make_loss(config: dict) -> torch.nn.Module:
    if config["training"]["loss"] != "SmoothL1Loss":
        raise ValueError(f"Unsupported loss: {config['training']['loss']}")
    return torch.nn.SmoothL1Loss(beta=float(config["training"]["smooth_l1_beta"]))


def make_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    if config["training"]["optimizer"] != "AdamW":
        raise ValueError(f"Unsupported optimizer: {config['training']['optimizer']}")
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )


def make_scheduler(optimizer: torch.optim.Optimizer, config: dict):
    if config["scheduler"]["name"] != "ReduceLROnPlateau":
        raise ValueError(f"Unsupported scheduler: {config['scheduler']['name']}")
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=config["scheduler"]["mode"],
        factor=float(config["scheduler"]["factor"]),
        patience=int(config["scheduler"]["patience"]),
        min_lr=float(config["scheduler"]["min_lr"]),
    )


def make_batches(config: dict, split: str, device: torch.device, shuffle: bool, return_raw_target: bool = False):
    return FastLSTMBatchBuilder(
        data_dir=config["paths"]["data_dir"],
        split=split,
        batch_size=int(config["training"]["batch_size"]),
        device=device,
        shuffle=shuffle,
        return_static=True,
        return_raw_target=return_raw_target,
        mmap_mode=config["data"].get("mmap_mode", "r"),
    )


def forward_batch(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return model(
        batch["x"],
        static_numeric=batch["static_numeric"],
        station_index=batch["station_idx"],
        district_id=batch["district_id"],
    )


def init_wandb(config: dict, metadata: dict, checkpoint_wandb_run_id: str | None):
    if not config["wandb"]["enabled"]:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb is enabled but not installed. Run `pip install -r requirements.txt`.") from exc

    wandb_run_id = config["wandb"].get("run_id") or checkpoint_wandb_run_id
    flattened = flatten_config(config)
    flattened.update(metadata)
    return wandb.init(
        entity=config["wandb"]["entity"],
        project=config["wandb"]["project"],
        name=config["wandb"]["name"],
        group=config["wandb"]["group"],
        job_type=config["wandb"]["job_type"],
        tags=config["project"]["tags"],
        notes=config["project"]["notes"],
        config=flattened,
        id=wandb_run_id,
        resume=config["wandb"]["resume"],
    )


def sync_global_step_with_wandb(global_step: int, wandb_run) -> int:
    if wandb_run is None:
        return global_step
    wandb_step = int(getattr(wandb_run, "step", 0) or 0)
    return max(global_step, wandb_step)


def smoke_test(
    model: torch.nn.Module,
    config: dict,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool,
    grad_scaler,
    batch_size: int,
) -> float:
    old_batch_size = int(config["training"]["batch_size"])
    config["training"]["batch_size"] = int(batch_size)
    batch = next(iter(make_batches(config, "train", device, shuffle=True)))
    config["training"]["batch_size"] = old_batch_size
    if not torch.isfinite(batch["x"]).all() or not torch.isfinite(batch["y"]).all():
        raise ValueError("Smoke test found non-finite input or target values.")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(use_amp):
        pred = forward_batch(model, batch)
        loss = loss_fn(pred, batch["y"])
    if not torch.isfinite(loss):
        raise ValueError("Smoke test produced a non-finite loss.")
    if grad_scaler is not None and use_amp:
        grad_scaler.scale(loss).backward()
    else:
        loss.backward()
    optimizer.zero_grad(set_to_none=True)
    return float(loss.item())


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    config: dict,
    split: str,
    loss_fn: torch.nn.Module,
    target_scalers: dict,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    batches = make_batches(config, split, device, shuffle=False, return_raw_target=True)
    metrics = RawCountMetricAccumulator()
    total_loss = 0.0
    total_samples = 0
    for batch_idx, batch in enumerate(tqdm(batches, desc=f"eval {split}", leave=False, total=len(batches))):
        if max_batches is not None and batch_idx >= max_batches:
            break
        pred = forward_batch(model, batch)
        loss = loss_fn(pred, batch["y"])
        batch_size = int(batch["y"].shape[0])
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        pred_raw = inverse_transform_targets(pred, target_scalers)
        metrics.update(pred_raw, batch["y_raw"])
    raw_metrics = metrics.compute()
    raw_metrics["loss"] = total_loss / max(total_samples, 1)
    return raw_metrics


def train_one_epoch(
    model: torch.nn.Module,
    config: dict,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_scaler,
    use_amp: bool,
    global_step: int,
    wandb_run,
    epoch: int,
) -> tuple[float, int]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    batches = make_batches(config, "train", device, shuffle=True)
    for batch in tqdm(batches, desc=f"train epoch {epoch}", leave=False, total=len(batches)):
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(use_amp):
            pred = forward_batch(model, batch)
            loss = loss_fn(pred, batch["y"])
        if grad_scaler is not None and use_amp:
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip_norm"]))
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip_norm"]))
            optimizer.step()

        batch_size = int(batch["y"].shape[0])
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        global_step += 1
        if wandb_run is not None and global_step % int(config["wandb"]["log_every_n_steps"]) == 0:
            wandb_run.log(
                {"train/loss": float(loss.item()), "train/lr": get_current_lr(optimizer), "train/epoch": epoch},
                step=global_step,
            )
    return total_loss / max(total_samples, 1), global_step


def build_metadata(config: dict) -> dict[str, Any]:
    data_dir = Path(config["paths"]["data_dir"])
    summary, feature_config, district_vocab = load_feature_metadata(data_dir)
    return {
        "architecture": config["model"]["architecture"],
        "data_dir": str(data_dir),
        "input_dim": len(feature_config["dynamic_feature_columns"]),
        "static_numeric_dim": len(feature_config["static_numeric_columns"]),
        "output_dim": len(feature_config["target_columns"]),
        "num_stations": int(summary["S"]),
        "num_districts": len(district_vocab),
        "window_offsets": feature_config["window_offsets"],
        "horizon": feature_config["horizon"],
        "train_sample_count": int(summary["samples_per_split"]["train"]),
        "val_sample_count": int(summary["samples_per_split"]["val"]),
        "eval_scale": "raw_count",
    }


def maybe_load_colab_wandb_key() -> None:
    try:
        from google.colab import userdata  # type: ignore

        wandb_key = userdata.get("WANDB_API_KEY")
        if wandb_key:
            os.environ["WANDB_API_KEY"] = wandb_key
    except Exception:
        return


def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    ensure_dirs(config["paths"]["checkpoint_dir"], config["paths"]["model_dir"], config["paths"]["log_dir"])
    set_seed(int(config["training"]["seed"]))
    device = select_device(config["training"]["device"])
    use_amp = bool(config["training"]["mixed_precision"]) and device.type == "cuda"

    data_dir = Path(config["paths"]["data_dir"])
    model = build_model(config, data_dir, device)
    loss_fn = make_loss(config)
    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(optimizer, config)
    grad_scaler = make_grad_scaler(use_amp)
    target_scalers = load_target_scalers(data_dir)

    if args.smoke_test:
        loss = smoke_test(model, config, loss_fn, optimizer, device, use_amp, grad_scaler, args.smoke_batch_size)
        print(f"Smoke test passed. loss={loss:.6f}")

    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    resume_path = resolve_resume_checkpoint(
        checkpoint_dir,
        config["resume"].get("checkpoint_path"),
        config["resume"].get("mode", "auto"),
    )
    checkpoint = None
    checkpoint_wandb_run_id = None
    start_epoch = 0
    global_step = 0
    best_metric = initial_best(config["checkpointing"]["mode"])
    best_epoch = -1
    epochs_without_improvement = 0
    if resume_path is not None and config["resume"]["enabled"]:
        print(f"Resuming from checkpoint: {resume_path}")
        checkpoint = load_checkpoint(resume_path, map_location=device)
        checkpoint_wandb_run_id = checkpoint.get("wandb_run_id")
        resume_mode = config["resume"].get("mode", "auto")
        if resume_mode == "auto":
            resume_mode = "full"
        restored = restore_checkpoint_state(
            checkpoint,
            model,
            optimizer,
            scheduler,
            grad_scaler,
            resume_mode=resume_mode,
            reset_optimizer=bool(config["resume"]["reset_optimizer"]),
            reset_scheduler=bool(config["resume"]["reset_scheduler"]),
        )
        start_epoch = int(restored["start_epoch"])
        global_step = int(restored["global_step"])
        if restored["best_metric"] is not None:
            best_metric = float(restored["best_metric"])
        best_epoch = int(restored["best_epoch"])
        epochs_without_improvement = int(restored["epochs_without_improvement"])

    maybe_load_colab_wandb_key()
    wandb_run = init_wandb(config, build_metadata(config), checkpoint_wandb_run_id)
    global_step = sync_global_step_with_wandb(global_step, wandb_run)

    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"
    max_epochs = int(config["training"]["max_epochs"])
    for epoch in range(start_epoch, max_epochs):
        train_loss, global_step = train_one_epoch(
            model, config, loss_fn, optimizer, device, grad_scaler, use_amp, global_step, wandb_run, epoch
        )
        val_metrics = evaluate_model(
            model,
            config,
            "val",
            loss_fn,
            target_scalers,
            device,
            max_batches=config["validation"]["max_val_batches"],
        )
        scheduler.step(val_metrics["total_mae"])
        monitor_value = float(val_metrics["total_mae"])
        improved = is_improvement(monitor_value, best_metric, config["checkpointing"]["mode"])
        if improved:
            best_metric = monitor_value
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        state = make_checkpoint(
            epoch,
            global_step,
            model,
            optimizer,
            scheduler,
            grad_scaler if use_amp else None,
            best_metric,
            best_epoch,
            epochs_without_improvement,
            config,
            wandb_run.id if wandb_run is not None else checkpoint_wandb_run_id,
            config["checkpointing"],
        )
        if config["checkpointing"]["save_last"]:
            save_checkpoint(last_path, state)
        if improved and config["checkpointing"]["save_best"]:
            save_checkpoint(best_path, state)

        payload = {
            "train/loss": train_loss,
            "train/lr": get_current_lr(optimizer),
            "train/epoch": epoch,
            "val/loss": val_metrics["loss"],
            "val/total_mae": val_metrics["total_mae"],
            "val/total_rmse": val_metrics["total_rmse"],
            "val/rental_mae": val_metrics["rental_mae"],
            "val/return_mae": val_metrics["return_mae"],
            "best_epoch": best_epoch,
            "best_val_total_mae": best_metric,
        }
        print(json.dumps(detach_metric_dict(payload), indent=2))
        if wandb_run is not None:
            wandb_run.log(payload, step=global_step)

        if config["early_stopping"]["enabled"] and epochs_without_improvement >= int(config["early_stopping"]["patience"]):
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}.")
            break

    final_metrics: dict[str, Any] = {
        "best_epoch": best_epoch,
        "best_val_total_mae": best_metric,
        "checkpoint_best_path": str(best_path),
        "checkpoint_last_path": str(last_path),
    }
    if config["testing"]["run_test_after_training"]:
        eval_checkpoint = best_path if config["testing"]["checkpoint"] == "best" else last_path
        if eval_checkpoint.exists():
            model.load_state_dict(load_checkpoint(eval_checkpoint, map_location=device)["model_state_dict"])
        test_split = "test_2025_winter" if (data_dir / "sample_index_test_2025_winter.npy").exists() else "test"
        test_metrics = evaluate_model(model, config, test_split, loss_fn, target_scalers, device)
        final_metrics.update({f"{test_split}_{key}": value for key, value in test_metrics.items()})
        if wandb_run is not None:
            wandb_run.log({f"{test_split}/{key}": value for key, value in test_metrics.items()}, step=global_step)

    model_path = Path(config["paths"]["model_dir"]) / f"{config['project']['run_name']}_best.pt"
    if best_path.exists():
        model.load_state_dict(load_checkpoint(best_path, map_location=device)["model_state_dict"])
    torch.save(model.state_dict(), model_path)
    final_metrics["final_model_path"] = str(model_path)
    write_json(Path(config["paths"]["log_dir"]) / "final_metrics.json", final_metrics)
    if wandb_run is not None:
        wandb_run.summary.update(final_metrics)
        wandb_run.finish()


if __name__ == "__main__":
    main()
