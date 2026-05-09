from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.data.tct_gat.tct_gat_dataset import FastTCTGATBatchBuilder
from src.training.lstm_training.checkpointing import load_checkpoint
from src.training.lstm_training.config import load_config
from src.training.lstm_training.utils import bool_from_string, ensure_dirs, select_device, write_json
from src.training.tct_gat_training.autoregressive_rollout import rollout_autoregressive
from src.training.tct_gat_training.metrics import RawCountMetricAccumulator, inverse_transform_targets, load_target_scalers
from src.training.tct_gat_training.train_tct_gat import build_model, evaluate_model, make_loss, maybe_load_colab_wandb_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a TCT-GAT1-AR checkpoint.")
    parser.add_argument("--config", type=Path, default=Path("configs/models/tct_gat/tct_gat1_ar.yaml"))
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--graph_dir", type=str, default=None)
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
        config={"data_dir": config["paths"]["data_dir"], "graph_dir": config["paths"]["graph_dir"], "split": split, "checkpoint_path": str(checkpoint_path)},
    )


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    cfg = json.loads(json.dumps(config))
    if args.data_dir is not None:
        cfg["paths"]["data_dir"] = args.data_dir
    if args.graph_dir is not None:
        cfg["paths"]["graph_dir"] = args.graph_dir
    if args.log_dir is not None:
        cfg["paths"]["log_dir"] = args.log_dir
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.device is not None:
        cfg["training"]["device"] = args.device
    if args.wandb_enabled is not None:
        cfg["wandb"]["enabled"] = args.wandb_enabled
    return cfg


def _to_tensor(array: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(np.array(array, copy=True), dtype=dtype, device=device)


def build_rollout_inputs(
    batches: FastTCTGATBatchBuilder,
    batch: dict[str, torch.Tensor],
    horizons: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    target_idx = batch["target_time_idx"].detach().cpu().numpy().astype(np.int64)
    min_offset = int(np.min(batches.window_offsets))
    history_offsets = np.arange(min_offset, 0, dtype=np.int64)
    future_offsets = np.arange(horizons, dtype=np.int64)
    history_idx = target_idx[:, None] + history_offsets[None, :]
    future_idx = target_idx[:, None] + future_offsets[None, :]
    if history_idx.min() < 0 or future_idx.max() >= batches.targets.shape[0]:
        return None
    rental_history = batches.rental_features[history_idx, :, :]
    return_history = batches.return_features[history_idx, :, :]
    future_target_time = batches.target_time_features[future_idx, :]
    future_weather = batches.future_weather_features[future_idx, :]
    y_scaled = batches.targets[future_idx, :, :]
    if batches.targets_raw is None:
        raise ValueError("targets_raw.npy is required for rollout metrics.")
    y_raw = batches.targets_raw[future_idx, :, :]
    return (
        _to_tensor(rental_history, device, torch.float32),
        _to_tensor(return_history, device, torch.float32),
        _to_tensor(future_target_time, device, torch.float32),
        _to_tensor(future_weather, device, torch.float32),
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
    device: torch.device,
    max_batches: int | None,
    horizons: int,
    monte_carlo_samples: int,
    weather_noise_eval: bool,
) -> dict[str, float]:
    model.eval()
    batches = FastTCTGATBatchBuilder(
        config["paths"]["data_dir"],
        split,
        int(config["training"]["batch_size"]),
        device=device,
        shuffle=False,
        return_raw_target=True,
        mmap_mode=config["data"].get("mmap_mode", "r"),
    )
    window_offsets = torch.as_tensor(batches.window_offsets, dtype=torch.long, device=device)
    metrics = RawCountMetricAccumulator(prefix="rollout")
    total_loss = 0.0
    total_batches = 0
    skipped = 0
    for batch_idx, batch in enumerate(tqdm(batches, desc=f"rollout eval {split}", leave=False, total=len(batches))):
        if max_batches is not None and batch_idx >= max_batches:
            break
        rollout_inputs = build_rollout_inputs(batches, batch, horizons, device)
        if rollout_inputs is None:
            skipped += 1
            continue
        rental_history, return_history, future_target_time, future_weather, y_scaled, y_raw = rollout_inputs
        predictions = []
        for _ in range(int(monte_carlo_samples)):
            predictions.append(
                rollout_autoregressive(
                    model=model,
                    initial_rental_history=rental_history,
                    initial_return_history=return_history,
                    future_target_time_features=future_target_time,
                    future_weather_features=future_weather,
                    static_numeric=batch["static_numeric"],
                    station_index=batch["station_index"],
                    district_id=batch["district_id"],
                    operation_type_id=batch.get("operation_type_id"),
                    window_offsets=window_offsets,
                    scalers={},
                    n_horizons=horizons,
                    weather_noise_config=config.get("weather_noise"),
                    apply_weather_noise_eval=weather_noise_eval,
                )
            )
        pred = predictions[0] if len(predictions) == 1 else torch.stack(predictions, dim=0).mean(dim=0)
        loss = loss_fn(pred, y_scaled)
        total_loss += float(loss.item())
        total_batches += 1
        metrics.update(inverse_transform_targets(pred, target_scalers), y_raw)
    out = metrics.compute()
    out["loss"] = total_loss / max(total_batches, 1)
    out["autoregressive_rollout"] = True
    out["rollout_horizons"] = float(horizons)
    out["monte_carlo_samples"] = float(monte_carlo_samples)
    out["weather_noise_eval"] = bool(weather_noise_eval)
    out["skipped_batches_without_future"] = float(skipped)
    return out


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    ensure_dirs(config["paths"]["log_dir"])
    device = select_device(config["training"]["device"])
    model = build_model(config, device)
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
            device,
            args.max_batches,
            args.rollout_horizons,
            args.monte_carlo_samples,
            args.weather_noise_eval,
        )
        output_name = f"{args.split}_rollout_metrics.json"
    else:
        metrics = evaluate_model(model, config, args.split, loss_fn, target_scalers, device, args.max_batches)
        output_name = f"{args.split}_metrics.json"
    metrics["checkpoint_path"] = str(args.checkpoint_path)
    metrics["split"] = args.split
    metrics["weather_mode"] = "noisy" if args.weather_noise_eval else "oracle"
    output_path = Path(config["paths"]["log_dir"]) / output_name
    write_json(output_path, metrics)
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {output_path}")

    wandb_run = init_eval_wandb(config, args.split, args.checkpoint_path)
    if wandb_run is not None:
        if args.autoregressive_rollout:
            wandb_run.log({f"rollout/{key}": value for key, value in metrics.items() if isinstance(value, (int, float, bool))})
            prefix = "weather/noisy" if args.weather_noise_eval else "weather/oracle"
            if "rollout_total_mae" in metrics:
                wandb_run.log({f"{prefix}_total_mae": metrics["rollout_total_mae"]})
        else:
            wandb_run.log({f"eval/{key}": value for key, value in metrics.items() if isinstance(value, (int, float, bool))})
        wandb_run.summary.update(metrics)
        wandb_run.finish()


if __name__ == "__main__":
    main()
