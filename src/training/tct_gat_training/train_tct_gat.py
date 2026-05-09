from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.data.tct_gat.tct_gat_dataset import FastTCTGATBatchBuilder, validate_graph_files
from src.models.tct_gat import TCTGAT1AR
from src.training.lstm_training.checkpointing import (
    initial_best,
    is_improvement,
    load_checkpoint,
    make_checkpoint,
    resolve_resume_checkpoint,
    restore_checkpoint_state,
    save_checkpoint,
)
from src.training.lstm_training.config import flatten_config, load_config
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
from src.training.tct_gat_training.metrics import RawCountMetricAccumulator, inverse_transform_targets, load_target_scalers
from src.training.tct_gat_training.weather_scenarios import apply_future_weather_noise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TCT-GAT1-AR.")
    parser.add_argument("--config", type=Path, default=Path("configs/models/tct_gat/tct_gat1_ar.yaml"))
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--graph_dir", type=str, default=None)
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
    parser.add_argument("--smoke_batch_size", type=int, default=1)
    return parser.parse_args()


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = json.loads(json.dumps(config))
    for key in ["data_dir", "graph_dir", "checkpoint_dir", "model_dir", "log_dir"]:
        value = getattr(args, key)
        if value is not None:
            cfg["paths"][key] = value
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        cfg["training"]["learning_rate"] = args.learning_rate
    if args.max_epochs is not None:
        cfg["training"]["max_epochs"] = args.max_epochs
    if args.wandb_enabled is not None:
        cfg["wandb"]["enabled"] = args.wandb_enabled
    if args.resume is not None:
        cfg["resume"]["enabled"] = True
        cfg["resume"]["checkpoint_path"] = "auto" if args.resume == "auto" else args.resume
        cfg["resume"]["mode"] = args.resume_mode or "auto"
    elif args.resume_mode is not None:
        cfg["resume"]["mode"] = args.resume_mode
    return cfg


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


def validate_config_guardrails(config: dict[str, Any]) -> None:
    if str(config["model"].get("architecture")) != "tct_gat1_ar":
        raise ValueError("TCT-GAT training requires model.architecture=tct_gat1_ar.")
    expected = {"input_dim": 5, "target_time_dim": 8, "future_weather_dim": 4, "output_dim": 2}
    for key, value in expected.items():
        if int(config["data"][key]) != value:
            raise ValueError(f"TCT-GAT requires data.{key}={value}, got {config['data'][key]}.")
    data_dir = Path(config["paths"]["data_dir"])
    graph_dir = Path(config["paths"]["graph_dir"])
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")
    if not graph_dir.exists():
        raise FileNotFoundError(f"graph_dir does not exist: {graph_dir}")
    feature_config = load_json(data_dir / "feature_config.json")
    if "net_demand" in json.dumps(feature_config, ensure_ascii=False):
        raise ValueError("net_demand must not appear in TCT-GAT feature_config.")
    rental_features = np.load(data_dir / "rental_features.npy", mmap_mode="r")
    validate_graph_files(graph_dir, rental_features.shape[1])


def build_model(config: dict[str, Any], device: torch.device) -> torch.nn.Module:
    data_dir = Path(config["paths"]["data_dir"])
    summary = load_json(data_dir / "dataset_summary.json")
    feature_config = load_json(data_dir / "feature_config.json")
    model_config = config["model"]
    model = TCTGAT1AR(
        graph_dir=config["paths"]["graph_dir"],
        input_dim=int(config["data"]["input_dim"]),
        target_time_dim=int(config["data"]["target_time_dim"]),
        future_weather_dim=int(config["data"]["future_weather_dim"]),
        output_dim=int(config["data"]["output_dim"]),
        num_stations=int(summary["num_stations"]),
        num_districts=len(feature_config["district_vocab"]),
        num_operation_types=len(feature_config["operation_type_vocab"]),
        static_numeric_dim=len(feature_config["static_numeric_columns"]),
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
        operation_type_embedding_dim=int(model_config["operation_type_embedding_dim"]),
        static_context_dim=int(model_config["static_context_dim"]),
        token_dim=int(model_config["token_dim"]),
        gat_layers=int(model_config["gat_layers"]),
        gat_heads=int(model_config["gat_heads"]),
        edge_embedding_dim=int(model_config["edge_embedding_dim"]),
        decoder_hidden_dims=model_config["decoder_hidden_dims"],
        dropout=float(model_config["dropout"]),
        gat_dropout=float(model_config["gat_dropout"]),
        attention_dropout=float(model_config["attention_dropout"]),
    )
    return model.to(device)


def make_loss(config: dict[str, Any]) -> torch.nn.Module:
    if config["training"]["loss"] != "SmoothL1Loss":
        raise ValueError(f"Unsupported loss: {config['training']['loss']}")
    return torch.nn.SmoothL1Loss(beta=float(config["training"]["smooth_l1_beta"]))


def make_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    if config["training"]["optimizer"] != "AdamW":
        raise ValueError(f"Unsupported optimizer: {config['training']['optimizer']}")
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )


def make_scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any]):
    if config["scheduler"]["name"] != "ReduceLROnPlateau":
        raise ValueError(f"Unsupported scheduler: {config['scheduler']['name']}")
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=config["scheduler"]["mode"],
        factor=float(config["scheduler"]["factor"]),
        patience=int(config["scheduler"]["patience"]),
        min_lr=float(config["scheduler"]["min_lr"]),
    )


def make_batches(config: dict[str, Any], split: str, device: torch.device, shuffle: bool, return_raw_target: bool = False):
    return FastTCTGATBatchBuilder(
        data_dir=config["paths"]["data_dir"],
        split=split,
        batch_size=int(config["training"]["batch_size"]),
        device=device,
        shuffle=shuffle,
        return_raw_target=return_raw_target,
        mmap_mode=config["data"].get("mmap_mode", "r"),
    )


def forward_batch(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return model(
        rental_seq=batch["rental_seq"],
        return_seq=batch["return_seq"],
        target_time_features=batch["target_time_features"],
        future_weather_features=batch["future_weather_features"],
        static_numeric=batch["static_numeric"],
        station_index=batch["station_index"],
        district_id=batch["district_id"],
        operation_type_id=batch.get("operation_type_id"),
    )


def maybe_load_colab_wandb_key() -> None:
    try:
        from google.colab import userdata  # type: ignore

        wandb_key = userdata.get("WANDB_API_KEY")
        if wandb_key:
            os.environ["WANDB_API_KEY"] = wandb_key
    except Exception:
        return


def init_wandb(config: dict[str, Any], metadata: dict[str, Any], checkpoint_wandb_run_id: str | None):
    if not config["wandb"]["enabled"]:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb is enabled but not installed. Run `pip install -r requirements.txt`.") from exc
    maybe_load_colab_wandb_key()
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
    return max(global_step, int(getattr(wandb_run, "step", 0) or 0))


def build_metadata(config: dict[str, Any]) -> dict[str, Any]:
    data_dir = Path(config["paths"]["data_dir"])
    summary = load_json(data_dir / "dataset_summary.json")
    graph_summary = load_json(Path(config["paths"]["graph_dir"]) / "graph_summary.json")
    return {
        "architecture": config["model"]["architecture"],
        "data_dir": str(data_dir),
        "graph_dir": config["paths"]["graph_dir"],
        "num_stations": int(summary["num_stations"]),
        "k_neighbors": int(graph_summary["k_neighbors"]),
        "edge_feature_dim": len(graph_summary["edge_feature_columns"]),
        "gat_layers": int(config["model"]["gat_layers"]),
        "gat_heads": int(config["model"]["gat_heads"]),
        "token_dim": int(config["model"]["token_dim"]),
        "batch_size": int(config["training"]["batch_size"]),
        "train_sample_count": int(summary["samples_per_split"]["train"]),
        "val_sample_count": int(summary["samples_per_split"]["val"]),
    }


def smoke_test(
    model: torch.nn.Module,
    config: dict[str, Any],
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    use_amp: bool,
    grad_scaler,
    batch_size: int,
    device: torch.device,
) -> float:
    old_batch = int(config["training"]["batch_size"])
    config["training"]["batch_size"] = int(batch_size)
    batch = next(iter(make_batches(config, "train", device, shuffle=True)))
    config["training"]["batch_size"] = old_batch
    for key in ["rental_seq", "return_seq", "y", "target_time_features", "future_weather_features"]:
        if not torch.isfinite(batch[key]).all():
            raise ValueError(f"Smoke test found non-finite values in {key}.")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(use_amp):
        pred = forward_batch(model, batch)
        if pred.shape != batch["y"].shape:
            raise ValueError(f"Smoke pred shape {tuple(pred.shape)} does not match y {tuple(batch['y'].shape)}.")
        loss = loss_fn(pred, batch["y"])
    if not torch.isfinite(loss):
        raise ValueError("Smoke test produced non-finite loss.")
    if grad_scaler is not None and use_amp:
        grad_scaler.scale(loss).backward()
    else:
        loss.backward()
    for name, buffer in model.named_buffers():
        if name.endswith("edge_attr_rr") or "edge_attr" in name or "neighbor_index" in name:
            if buffer.requires_grad:
                raise ValueError(f"Graph buffer {name} unexpectedly requires grad.")
    optimizer.zero_grad(set_to_none=True)
    return float(loss.item())


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    config: dict[str, Any],
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
    total_batches = 0
    for batch_idx, batch in enumerate(tqdm(batches, desc=f"eval {split}", leave=False, total=len(batches))):
        if max_batches is not None and batch_idx >= max_batches:
            break
        pred = forward_batch(model, batch)
        loss = loss_fn(pred, batch["y"])
        total_loss += float(loss.item())
        total_batches += 1
        metrics.update(inverse_transform_targets(pred, target_scalers), batch["y_raw"])
    out = metrics.compute()
    out["loss"] = total_loss / max(total_batches, 1)
    return out


def train_one_epoch(
    model: torch.nn.Module,
    config: dict[str, Any],
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    grad_scaler,
    use_amp: bool,
    device: torch.device,
    global_step: int,
    wandb_run,
    epoch: int,
) -> tuple[float, int]:
    model.train()
    batches = make_batches(config, "train", device, shuffle=True)
    total_loss = 0.0
    total_batches = 0
    for batch in tqdm(batches, desc=f"train epoch {epoch}", leave=False, total=len(batches)):
        optimizer.zero_grad(set_to_none=True)
        if bool(config.get("weather_noise", {}).get("enabled_train", False)):
            batch["future_weather_features"] = apply_future_weather_noise(
                batch["future_weather_features"],
                config.get("weather_noise", {}),
                enabled=True,
            )
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
        total_loss += float(loss.item())
        total_batches += 1
        global_step += 1
        if wandb_run is not None and global_step % int(config["wandb"]["log_every_n_steps"]) == 0:
            wandb_run.log({"train/loss": float(loss.item()), "train/lr": get_current_lr(optimizer), "train/epoch": epoch}, step=global_step)
    return total_loss / max(total_batches, 1), global_step


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    validate_config_guardrails(config)
    ensure_dirs(config["paths"]["checkpoint_dir"], config["paths"]["model_dir"], config["paths"]["log_dir"])
    set_seed(int(config["training"]["seed"]))
    device = select_device(config["training"]["device"])
    use_amp = bool(config["training"]["mixed_precision"]) and device.type == "cuda"
    model = build_model(config, device)
    loss_fn = make_loss(config)
    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(optimizer, config)
    grad_scaler = make_grad_scaler(use_amp)
    target_scalers = load_target_scalers(config["paths"]["data_dir"])

    if args.smoke_test:
        loss = smoke_test(model, config, loss_fn, optimizer, use_amp, grad_scaler, args.smoke_batch_size, device)
        print(f"Smoke test passed. loss={loss:.6f}")

    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    resume_path = resolve_resume_checkpoint(checkpoint_dir, config["resume"].get("checkpoint_path"), config["resume"].get("mode", "auto"))
    checkpoint_wandb_run_id = None
    start_epoch = 0
    global_step = 0
    best_metric = initial_best(config["checkpointing"]["mode"])
    best_epoch = -1
    epochs_without_improvement = 0
    if resume_path is not None and config["resume"]["enabled"]:
        checkpoint = load_checkpoint(resume_path, map_location=device)
        checkpoint_wandb_run_id = checkpoint.get("wandb_run_id")
        resume_mode = "full" if config["resume"].get("mode", "auto") == "auto" else config["resume"].get("mode", "auto")
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

    wandb_run = init_wandb(config, build_metadata(config), checkpoint_wandb_run_id)
    global_step = sync_global_step_with_wandb(global_step, wandb_run)
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"

    for epoch in range(start_epoch, int(config["training"]["max_epochs"])):
        train_loss, global_step = train_one_epoch(model, config, loss_fn, optimizer, grad_scaler, use_amp, device, global_step, wandb_run, epoch)
        val_metrics = evaluate_model(model, config, "val", loss_fn, target_scalers, device, config["validation"]["max_val_batches"])
        scheduler.step(val_metrics["total_mae"])
        monitor = float(val_metrics["total_mae"])
        improved = is_improvement(monitor, best_metric, config["checkpointing"]["mode"])
        if improved:
            best_metric = monitor
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
            "val/rental_rmse": val_metrics["rental_rmse"],
            "val/return_mae": val_metrics["return_mae"],
            "val/return_rmse": val_metrics["return_rmse"],
            "best_epoch": best_epoch,
            "best_val_total_mae": best_metric,
        }
        print(json.dumps(detach_metric_dict(payload), indent=2))
        if wandb_run is not None:
            wandb_run.log(payload, step=global_step)
        if config["early_stopping"]["enabled"] and epochs_without_improvement >= int(config["early_stopping"]["patience"]):
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}.")
            break

    final_metrics: dict[str, Any] = {"best_epoch": best_epoch, "best_val_total_mae": best_metric, "checkpoint_best_path": str(best_path), "checkpoint_last_path": str(last_path)}
    ran_training = int(config["training"]["max_epochs"]) > start_epoch
    if ran_training and config["testing"]["run_test_after_training"]:
        eval_checkpoint = best_path if config["testing"]["checkpoint"] == "best" else last_path
        if eval_checkpoint.exists():
            model.load_state_dict(load_checkpoint(eval_checkpoint, map_location=device)["model_state_dict"])
        test_split = "test_2025_winter"
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
