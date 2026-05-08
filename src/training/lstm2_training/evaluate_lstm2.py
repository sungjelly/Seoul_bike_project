from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.data.lstm2.lstm2_dataset import FastLSTMBatchBuilder
from src.training.lstm_training.checkpointing import load_checkpoint
from src.training.lstm_training.config import load_config
from src.training.lstm_training.utils import bool_from_string, ensure_dirs, select_device, write_json
from src.training.lstm2_training.autoregressive_rollout import rollout_autoregressive
from src.training.lstm2_training.metrics import RawCountMetricAccumulator, inverse_transform_targets, load_target_scalers
from src.training.lstm2_training.train_lstm2 import (
    build_model,
    evaluate_model,
    is_v2_architecture,
    make_loss,
    maybe_load_colab_wandb_key,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Seoul bike LSTM2 checkpoint.")
    parser.add_argument("--config", type=Path, default=Path("configs/models/lstm2/tts_lstm2.yaml"))
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=Path, required=True)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="test_2025_winter")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--wandb_enabled", type=bool_from_string, default=None)
    parser.add_argument("--autoregressive_rollout", type=bool_from_string, default=False)
    parser.add_argument("--rollout_horizons", type=int, default=8)
    parser.add_argument("--monte_carlo_samples", type=int, default=1)
    parser.add_argument("--weather_noise_eval", type=bool_from_string, default=False)
    return parser.parse_args()


def init_eval_wandb(config: dict, split: str, checkpoint_path: Path):
    if not config["wandb"]["enabled"]:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb is enabled but not installed. Run `pip install -r requirements.txt`.") from exc
    maybe_load_colab_wandb_key()
    return wandb.init(
        entity=config["wandb"]["entity"],
        project=config["wandb"]["project"],
        name=f"{config['wandb']['name']}_eval_{split}",
        group=config["wandb"]["group"],
        job_type="eval",
        config={
            "data_dir": config["paths"]["data_dir"],
            "split": split,
            "checkpoint_path": str(checkpoint_path),
        },
    )


def load_full_scalers(data_dir: str | Path) -> dict:
    scalers_path = Path(data_dir) / "scalers.json"
    with scalers_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _to_tensor(array: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(np.array(array, copy=True), dtype=dtype, device=device)


def build_rollout_inputs(
    batches: FastLSTMBatchBuilder,
    batch: dict[str, torch.Tensor],
    n_horizons: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if n_horizons <= 0 or n_horizons > batches.targets.shape[-2]:
        raise ValueError(f"rollout_horizons must be in [1, {batches.targets.shape[-2]}], got {n_horizons}.")

    target_time_idx = batch["target_time_idx"].detach().cpu().numpy().astype(np.int64)
    station_idx = batch["station_idx"].detach().cpu().numpy().astype(np.int64)
    min_offset = int(np.min(batches.window_offsets))
    history_offsets = np.arange(min_offset, 0, dtype=np.int64)
    future_offsets = np.arange(n_horizons, dtype=np.int64)

    history_time_idx = target_time_idx[:, None] + history_offsets[None, :]
    future_time_idx = target_time_idx[:, None] + future_offsets[None, :]
    if history_time_idx.min() < 0 or future_time_idx.max() >= batches.dynamic_features.shape[0]:
        raise ValueError("Rollout sample requires history or future times outside dynamic_features.npy.")

    initial_dynamic_history = batches.dynamic_features[history_time_idx, station_idx[:, None], :]
    future_target_time_features = batches.target_time_features[future_time_idx, :]
    future_weather_features = batches.dynamic_features[future_time_idx, station_idx[:, None], :][:, :, [3, 4, 5, 6]]
    y_scaled = batches.targets[target_time_idx, station_idx, :n_horizons, :]
    if batches.targets_raw is None:
        raise ValueError("targets_raw.npy is required for autoregressive rollout metrics.")
    y_raw = batches.targets_raw[target_time_idx, station_idx, :n_horizons, :]

    return (
        _to_tensor(initial_dynamic_history, device, torch.float32),
        _to_tensor(future_target_time_features, device, torch.float32),
        _to_tensor(future_weather_features, device, torch.float32),
        _to_tensor(y_scaled, device, torch.float32),
        _to_tensor(y_raw, device, torch.float32),
    )


@torch.no_grad()
def evaluate_autoregressive_rollout(
    model: torch.nn.Module,
    config: dict,
    split: str,
    loss_fn: torch.nn.Module,
    target_scalers: dict,
    full_scalers: dict,
    device: torch.device,
    max_batches: int | None,
    n_horizons: int,
    monte_carlo_samples: int,
    weather_noise_eval: bool,
) -> dict[str, float]:
    if not is_v2_architecture(config):
        raise ValueError("--autoregressive_rollout is only supported for architecture=tts_lstm2_v2.")
    if monte_carlo_samples <= 0:
        raise ValueError(f"monte_carlo_samples must be positive, got {monte_carlo_samples}.")

    model.eval()
    batches = FastLSTMBatchBuilder(
        data_dir=config["paths"]["data_dir"],
        split=split,
        batch_size=int(config["training"]["batch_size"]),
        device=device,
        shuffle=False,
        return_static=True,
        return_raw_target=True,
        mmap_mode=config["data"].get("mmap_mode", "r"),
        one_step_target=True,
        target_horizon_index=0,
        return_future_weather=True,
    )
    window_offsets = torch.as_tensor(batches.window_offsets, dtype=torch.long, device=device)
    metrics = RawCountMetricAccumulator()
    total_loss = 0.0
    total_samples = 0

    for batch_idx, batch in enumerate(tqdm(batches, desc=f"rollout eval {split}", leave=False, total=len(batches))):
        if max_batches is not None and batch_idx >= max_batches:
            break
        (
            initial_dynamic_history,
            future_target_time_features,
            future_weather_features,
            y_scaled,
            y_raw,
        ) = build_rollout_inputs(batches, batch, n_horizons, device)

        predictions = []
        for _ in range(monte_carlo_samples):
            predictions.append(
                rollout_autoregressive(
                    model=model,
                    initial_dynamic_history=initial_dynamic_history,
                    future_target_time_features=future_target_time_features,
                    future_weather_features=future_weather_features,
                    static_numeric=batch["static_numeric"],
                    station_index=batch["station_idx"],
                    district_id=batch["district_id"],
                    operation_type_id=batch.get("operation_type_id"),
                    window_offsets=window_offsets,
                    scalers=full_scalers,
                    n_horizons=n_horizons,
                    weather_noise_config=config.get("weather_noise"),
                    apply_weather_noise_eval=weather_noise_eval,
                )
            )
        pred = predictions[0] if len(predictions) == 1 else torch.stack(predictions, dim=0).mean(dim=0)
        loss = loss_fn(pred, y_scaled)
        batch_size = int(y_scaled.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        pred_raw = inverse_transform_targets(pred, target_scalers)
        metrics.update(pred_raw, y_raw)

    raw_metrics = metrics.compute()
    raw_metrics["loss"] = total_loss / max(total_samples, 1)
    raw_metrics["autoregressive_rollout"] = True
    raw_metrics["rollout_horizons"] = float(n_horizons)
    raw_metrics["monte_carlo_samples"] = float(monte_carlo_samples)
    raw_metrics["weather_noise_eval"] = bool(weather_noise_eval)
    return raw_metrics


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.data_dir is not None:
        config["paths"]["data_dir"] = args.data_dir
    if args.log_dir is not None:
        config["paths"]["log_dir"] = args.log_dir
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.device is not None:
        config["training"]["device"] = args.device
    if args.wandb_enabled is not None:
        config["wandb"]["enabled"] = args.wandb_enabled

    sample_index_path = Path(config["paths"]["data_dir"]) / f"sample_index_{args.split}.npy"
    if not sample_index_path.exists():
        raise FileNotFoundError(f"Split sample index does not exist: {sample_index_path}")

    ensure_dirs(config["paths"]["log_dir"])
    device = select_device(config["training"]["device"])
    model = build_model(config, Path(config["paths"]["data_dir"]), device)
    checkpoint = load_checkpoint(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    loss_fn = make_loss(config)
    target_scalers = load_target_scalers(config["paths"]["data_dir"])

    if args.autoregressive_rollout:
        metrics = evaluate_autoregressive_rollout(
            model,
            config,
            args.split,
            loss_fn,
            target_scalers,
            load_full_scalers(config["paths"]["data_dir"]),
            device,
            max_batches=args.max_batches,
            n_horizons=args.rollout_horizons,
            monte_carlo_samples=args.monte_carlo_samples,
            weather_noise_eval=args.weather_noise_eval,
        )
    else:
        metrics = evaluate_model(
            model,
            config,
            args.split,
            loss_fn,
            target_scalers,
            device,
            max_batches=args.max_batches,
        )
    metrics["checkpoint_path"] = str(args.checkpoint_path)
    metrics["split"] = args.split
    output_path = Path(config["paths"]["log_dir"]) / f"{args.split}_metrics.json"
    write_json(output_path, metrics)
    print(metrics)
    print(f"Saved metrics to {output_path}")

    wandb_run = init_eval_wandb(config, args.split, args.checkpoint_path)
    if wandb_run is not None:
        wandb_run.log({f"eval/{key}": value for key, value in metrics.items() if isinstance(value, (int, float))})
        wandb_run.summary.update(metrics)
        wandb_run.finish()


if __name__ == "__main__":
    main()
