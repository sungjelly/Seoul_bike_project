from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.training.lstm_training.metrics import RawCountMetricAccumulator, inverse_transform_targets


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    target_scalers: dict,
    device: torch.device,
    max_batches: int | None = None,
    desc: str = "eval",
) -> dict[str, float]:
    model.eval()
    metrics = RawCountMetricAccumulator()
    total_loss = 0.0
    total_samples = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=desc, leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x_seq = batch["x_seq"].to(device, non_blocking=True)
        static_numeric = batch["static_numeric"].to(device, non_blocking=True)
        station_index = batch["station_index"].to(device, non_blocking=True)
        district_id = batch["district_id"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        pred = model(x_seq, static_numeric, station_index, district_id)
        loss = loss_fn(pred, target)

        batch_size = int(target.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

        pred_raw = inverse_transform_targets(pred, target_scalers)
        metrics.update(pred_raw, batch["target_raw"])

    raw_metrics = metrics.compute()
    raw_metrics["loss"] = total_loss / max(total_samples, 1)
    return raw_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained baseline LSTM checkpoint.")
    parser.add_argument("--config", type=Path, default=Path("configs/lstm_baseline.yaml"))
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=Path, required=True)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    from src.training.lstm_training.checkpointing import load_checkpoint
    from src.training.lstm_training.config import load_config
    from src.training.lstm_training.metrics import load_target_scalers
    from src.training.lstm_training.train_lstm import build_model, make_dataloader, make_loss
    from src.training.lstm_training.utils import ensure_dirs, select_device, write_json
    from src.data.lstm_dataset import LSTMBaselineDataset

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

    data_dir = Path(config["paths"]["data_dir"])
    log_dir = Path(config["paths"]["log_dir"])
    ensure_dirs(log_dir)
    device = select_device(config["training"]["device"])

    dataset = LSTMBaselineDataset(
        data_dir=data_dir,
        split=args.split,
        sequence_length=int(config["data"]["sequence_length"]),
        horizon=int(config["data"]["horizon"]),
        mmap_mode=config["data"]["mmap_mode"],
    )
    dataloader = make_dataloader(dataset, config, shuffle=False)

    model = build_model(config, data_dir, device)
    checkpoint = load_checkpoint(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    loss_fn = make_loss(config)
    target_scalers = load_target_scalers(data_dir)

    metrics = evaluate_model(
        model,
        dataloader,
        loss_fn,
        target_scalers,
        device,
        max_batches=args.max_batches,
        desc=f"evaluate {args.split}",
    )
    metrics["checkpoint_path"] = str(args.checkpoint_path)
    metrics["split"] = args.split
    output_path = log_dir / f"{args.split}_metrics.json"
    write_json(output_path, metrics)
    print(metrics)
    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()

