from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.data.lstm_dataset import LSTMBaselineDataset
from src.models.baseline_lstm import BaselineLSTM
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
from src.training.lstm_training.evaluate import evaluate_model
from src.training.lstm_training.metrics import load_target_scalers
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
    parser = argparse.ArgumentParser(description="Train the Seoul bike baseline LSTM.")
    parser.add_argument("--config", type=Path, default=Path("configs/lstm_baseline.yaml"))
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
    return parser.parse_args()


def collate_lstm_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "x_seq": torch.stack([item["x_seq"] for item in batch]),
        "static_numeric": torch.stack([item["static_numeric"] for item in batch]),
        "station_index": torch.stack([item["station_index"] for item in batch]),
        "district_id": torch.stack([item["district_id"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "target_raw": torch.stack([item["target_raw"] for item in batch]),
        "target_timestamp": [item["target_timestamp"] for item in batch],
    }


def make_dataloader(dataset, config: dict, shuffle: bool) -> DataLoader:
    num_workers = int(config["data"]["num_workers"])
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=bool(config["data"]["pin_memory"]),
        persistent_workers=bool(config["data"]["persistent_workers"]) and num_workers > 0,
        collate_fn=collate_lstm_batch,
    )


def build_model(config: dict, data_dir: Path, device: torch.device) -> BaselineLSTM:
    summary = load_json(data_dir / "dataset_summary.json")
    district_vocab = load_json(data_dir / "district_vocab.json")
    model = BaselineLSTM(
        input_dim=int(config["data"]["input_dim"]),
        static_numeric_dim=int(config["data"]["static_numeric_dim"]),
        output_dim=int(config["data"]["output_dim"]),
        num_stations=int(summary["S"]),
        num_districts=len(district_vocab),
        hidden_dim=int(config["model"]["hidden_dim"]),
        num_layers=int(config["model"]["num_layers"]),
        station_embedding_dim=int(config["model"]["station_embedding_dim"]),
        district_embedding_dim=int(config["model"]["district_embedding_dim"]),
        mlp_hidden_dim=int(config["model"]["mlp_hidden_dim"]),
        dropout=float(config["model"]["dropout"]),
    )
    return model.to(device)


def make_loss(config: dict) -> torch.nn.Module:
    if config["training"]["loss"] != "SmoothL1Loss":
        raise ValueError(f"Unsupported loss: {config['training']['loss']}")
    # SmoothL1 is used in transformed/scaled target space because it is less
    # sensitive to large demand spikes than MSE while remaining smooth near zero.
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


def set_optimizer_hparams(optimizer: torch.optim.Optimizer, config: dict) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(config["training"]["learning_rate"])
        group["weight_decay"] = float(config["training"]["weight_decay"])


def init_wandb(config: dict, metadata: dict, checkpoint_wandb_run_id: str | None):
    if not config["wandb"]["enabled"]:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "wandb is enabled but not installed. Install it with `pip install wandb`, "
            "or run with `--wandb_enabled false`."
        ) from exc

    wandb_run_id = config["wandb"].get("run_id") or checkpoint_wandb_run_id
    flattened = flatten_config(config)
    flattened.update(metadata)
    try:
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
    except Exception as exc:
        raise RuntimeError(
            "W&B initialization failed. Run `wandb.login()` first or set WANDB_API_KEY "
            "as an environment variable or Colab Secret. No API key should be saved in config files."
        ) from exc


def sync_global_step_with_wandb(global_step: int, wandb_run) -> int:
    """Keep resumed W&B logs monotonic.

    W&B will ignore logs sent to a step lower than the current resumed run step.
    This can happen when Colab disconnects after W&B received step logs but
    before last.pt was overwritten. We advance the local logging counter instead
    of trying to write out-of-order history.
    """
    if wandb_run is None:
        return global_step
    wandb_step = int(getattr(wandb_run, "step", 0) or 0)
    if wandb_step > global_step:
        print(
            f"W&B resumed at step {wandb_step}, but checkpoint global_step is {global_step}. "
            f"Advancing local global_step to {wandb_step} to keep W&B steps monotonic."
        )
        return wandb_step
    return global_step


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_scaler,
    use_amp: bool,
    grad_clip_norm: float,
    global_step: int,
    wandb_run,
    log_every_n_steps: int,
    epoch: int,
) -> tuple[float, int]:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in tqdm(dataloader, desc=f"train epoch {epoch}", leave=False):
        x_seq = batch["x_seq"].to(device, non_blocking=True)
        static_numeric = batch["static_numeric"].to(device, non_blocking=True)
        station_index = batch["station_index"].to(device, non_blocking=True)
        district_id = batch["district_id"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(use_amp):
            pred = model(x_seq, static_numeric, station_index, district_id)
            loss = loss_fn(pred, target)

        if grad_scaler is not None and use_amp:
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        batch_size = int(target.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        global_step += 1

        if wandb_run is not None and global_step % log_every_n_steps == 0:
            wandb_run.log(
                {
                    "train/loss": float(loss.item()),
                    "train/lr": get_current_lr(optimizer),
                    "train/epoch": epoch,
                    "train/global_step": global_step,
                },
                step=global_step,
            )

    return total_loss / max(total_samples, 1), global_step


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


def load_datasets(config: dict):
    data_dir = Path(config["paths"]["data_dir"])
    kwargs = {
        "data_dir": data_dir,
        "sequence_length": int(config["data"]["sequence_length"]),
        "horizon": int(config["data"]["horizon"]),
        "mmap_mode": config["data"]["mmap_mode"],
    }
    return (
        LSTMBaselineDataset(split="train", **kwargs),
        LSTMBaselineDataset(split="val", **kwargs),
        LSTMBaselineDataset(split="test", **kwargs),
    )


def build_metadata(config: dict, train_dataset, val_dataset, test_dataset) -> dict:
    data_dir = Path(config["paths"]["data_dir"])
    summary = load_json(data_dir / "dataset_summary.json")
    scalers = load_json(data_dir / "scalers.json")
    district_vocab = load_json(data_dir / "district_vocab.json")
    return {
        "architecture": config["model"]["architecture"],
        "data_dir": str(data_dir),
        "sequence_length": config["data"]["sequence_length"],
        "horizon": config["data"]["horizon"],
        "input_dim": config["data"]["input_dim"],
        "static_numeric_dim": config["data"]["static_numeric_dim"],
        "output_dim": config["data"]["output_dim"],
        "num_stations": int(summary["S"]),
        "num_districts": len(district_vocab),
        "hidden_dim": config["model"]["hidden_dim"],
        "num_layers": config["model"]["num_layers"],
        "station_embedding_dim": config["model"]["station_embedding_dim"],
        "district_embedding_dim": config["model"]["district_embedding_dim"],
        "mlp_hidden_dim": config["model"]["mlp_hidden_dim"],
        "dropout": config["model"]["dropout"],
        "loss": config["training"]["loss"],
        "smooth_l1_beta": config["training"]["smooth_l1_beta"],
        "optimizer": config["training"]["optimizer"],
        "learning_rate": config["training"]["learning_rate"],
        "weight_decay": config["training"]["weight_decay"],
        "batch_size": config["training"]["batch_size"],
        "max_epochs": config["training"]["max_epochs"],
        "gradient_clip_norm": config["training"]["gradient_clip_norm"],
        "mixed_precision": config["training"]["mixed_precision"],
        "scheduler": config["scheduler"],
        "early_stopping_patience": config["early_stopping"]["patience"],
        "train_sample_count": len(train_dataset),
        "val_sample_count": len(val_dataset),
        "test_sample_count": len(test_dataset),
        "target_transform": scalers["target"],
        "eval_scale": "raw_count",
    }


def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    model_dir = Path(config["paths"]["model_dir"])
    log_dir = Path(config["paths"]["log_dir"])
    ensure_dirs(checkpoint_dir, model_dir, log_dir)

    set_seed(int(config["training"]["seed"]))
    device = select_device(config["training"]["device"])
    use_amp = bool(config["training"]["mixed_precision"]) and device.type == "cuda"

    train_dataset, val_dataset, test_dataset = load_datasets(config)
    train_loader = make_dataloader(train_dataset, config, shuffle=True)
    val_loader = make_dataloader(val_dataset, config, shuffle=False)
    test_loader = make_dataloader(test_dataset, config, shuffle=False)

    data_dir = Path(config["paths"]["data_dir"])
    model = build_model(config, data_dir, device)
    loss_fn = make_loss(config)
    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(optimizer, config)
    grad_scaler = make_grad_scaler(use_amp)
    target_scalers = load_target_scalers(data_dir)

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
        if resume_mode == "model_optimizer":
            set_optimizer_hparams(optimizer, config)
        start_epoch = int(restored["start_epoch"])
        global_step = int(restored["global_step"])
        if restored["best_metric"] is not None:
            best_metric = float(restored["best_metric"])
        best_epoch = int(restored["best_epoch"])
        epochs_without_improvement = int(restored["epochs_without_improvement"])
        print(
            f"Loaded checkpoint with resume_mode={resume_mode}. "
            f"Continuing at epoch {start_epoch}, global_step {global_step}."
        )
    else:
        print("No checkpoint loaded. Starting a fresh training run.")

    metadata = build_metadata(config, train_dataset, val_dataset, test_dataset)
    wandb_run = init_wandb(config, metadata, checkpoint_wandb_run_id)
    global_step = sync_global_step_with_wandb(global_step, wandb_run)
    if wandb_run is not None and config["wandb"]["watch_model"]:
        wandb_run.watch(model)

    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"
    final_model_path = model_dir / f"{config['project']['run_name']}_best.pt"

    max_epochs = int(config["training"]["max_epochs"])
    for epoch in range(start_epoch, max_epochs):
        train_loss, global_step = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            grad_scaler,
            use_amp,
            float(config["training"]["gradient_clip_norm"]),
            global_step,
            wandb_run,
            int(config["wandb"]["log_every_n_steps"]),
            epoch,
        )

        # Validation metrics are inverse-transformed to raw counts because the
        # actual decision task is station-level bike demand, not scaled loss.
        val_metrics = evaluate_model(
            model,
            val_loader,
            loss_fn,
            target_scalers,
            device,
            max_batches=config["validation"]["max_val_batches"],
            desc=f"val epoch {epoch}",
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

        wandb_run_id = wandb_run.id if wandb_run is not None else checkpoint_wandb_run_id
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
            wandb_run_id,
            config["checkpointing"],
        )
        # Only last.pt and best.pt are saved by default. last.pt supports Colab
        # crash recovery; best.pt is the deploy/evaluate checkpoint.
        if config["checkpointing"]["save_last"]:
            save_checkpoint(last_path, state)
        if improved and config["checkpointing"]["save_best"]:
            save_checkpoint(best_path, state)

        log_payload = {
            "train/loss": train_loss,
            "train/lr": get_current_lr(optimizer),
            "train/epoch": epoch,
            "train/global_step": global_step,
            "val/loss": val_metrics["loss"],
            "val/rental_mae": val_metrics["rental_mae"],
            "val/rental_rmse": val_metrics["rental_rmse"],
            "val/return_mae": val_metrics["return_mae"],
            "val/return_rmse": val_metrics["return_rmse"],
            "val/net_demand_mae": val_metrics["net_demand_mae"],
            "val/net_demand_rmse": val_metrics["net_demand_rmse"],
            "val/total_mae": val_metrics["total_mae"],
            "val/total_rmse": val_metrics["total_rmse"],
            "best_epoch": best_epoch,
            "best_val_total_mae": best_metric,
        }
        print(json.dumps(detach_metric_dict(log_payload), indent=2))
        if wandb_run is not None:
            wandb_run.log(log_payload, step=global_step)

        if (
            config["early_stopping"]["enabled"]
            and epochs_without_improvement >= int(config["early_stopping"]["patience"])
        ):
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}.")
            break

    final_metrics: dict[str, Any] = {
        "best_epoch": best_epoch,
        "best_val_total_mae": best_metric,
        "checkpoint_best_path": str(best_path),
        "checkpoint_last_path": str(last_path),
        "final_model_path": str(final_model_path),
    }

    eval_checkpoint_path = best_path if config["testing"]["checkpoint"] == "best" else last_path
    if eval_checkpoint_path.exists():
        checkpoint = load_checkpoint(eval_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    if config["testing"]["run_test_after_training"]:
        test_metrics = evaluate_model(
            model,
            test_loader,
            loss_fn,
            target_scalers,
            device,
            desc="test",
        )
        final_metrics.update({f"test_{key}": value for key, value in test_metrics.items()})
        if wandb_run is not None:
            wandb_run.log(
                {
                    "test/rental_mae": test_metrics["rental_mae"],
                    "test/rental_rmse": test_metrics["rental_rmse"],
                    "test/return_mae": test_metrics["return_mae"],
                    "test/return_rmse": test_metrics["return_rmse"],
                    "test/net_demand_mae": test_metrics["net_demand_mae"],
                    "test/net_demand_rmse": test_metrics["net_demand_rmse"],
                    "test/total_mae": test_metrics["total_mae"],
                    "test/total_rmse": test_metrics["total_rmse"],
                },
                step=global_step,
            )

    torch.save(model.state_dict(), final_model_path)
    write_json(log_dir / "final_metrics.json", final_metrics)

    if wandb_run is not None:
        wandb_run.log(
            {
                "best_epoch": best_epoch,
                "best_val_total_mae": best_metric,
                "checkpoint_best_path": str(best_path),
                "checkpoint_last_path": str(last_path),
                "final_model_path": str(final_model_path),
            },
            step=global_step,
        )
        if config["wandb"]["log_model"]:
            wandb_run.save(str(final_model_path))
        wandb_run.finish()


if __name__ == "__main__":
    main()

