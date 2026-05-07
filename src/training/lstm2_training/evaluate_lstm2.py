from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.training.lstm_training.checkpointing import load_checkpoint
from src.training.lstm_training.config import load_config
from src.training.lstm_training.utils import bool_from_string, ensure_dirs, select_device, write_json
from src.training.lstm2_training.metrics import load_target_scalers
from src.training.lstm2_training.train_lstm2 import build_model, evaluate_model, make_loss, maybe_load_colab_wandb_key


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
