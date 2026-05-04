from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from src.data.lstm_baseline.lstm_dataset import resolve_array_dir
from src.training.lstm_training.utils import bool_from_string


BASELINES = {
    "zero": {"history": 0},
    "previous_window": {"history": 1},
    "same_time_yesterday": {"history": 48},
    "same_time_last_week": {"history": 336},
    "recent_mean_4h": {"history": 8},
    "recent_mean_24h": {"history": 48},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate naive raw-count demand baselines.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/lstm_processed/lstm_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/naive_baseline"))
    parser.add_argument("--splits", nargs="+", default=["val", "test_2025_winter", "test_2024_april_june"])
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--save-local", type=bool_from_string, default=False)
    parser.add_argument("--wandb-enabled", type=bool_from_string, default=True)
    parser.add_argument("--wandb-entity", type=str, default="sungjelly-kaist-digital-humanities-and-social-sciences-g")
    parser.add_argument("--wandb-project", type=str, default="Seoul_Bike_Project")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-job-type", type=str, default="naive_baseline")
    return parser.parse_args()


def load_split_indices(data_dir: Path) -> dict[str, np.ndarray]:
    splits_path = data_dir / "splits.json"
    with splits_path.open("r", encoding="utf-8") as f:
        splits = json.load(f)
    split_indices = {}
    for split_name, metadata in splits.items():
        sample_index_file = metadata.get("sample_index_file", f"sample_index_{split_name}.npy")
        sample_index = np.load(data_dir / sample_index_file, mmap_mode="r")
        split_indices[split_name] = np.unique(sample_index[:, 0].astype(np.int64))
    return split_indices


def load_source_ranges(data_dir: Path) -> list[tuple[int, int]]:
    path = data_dir / "source_boundaries.json"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        boundaries = json.load(f)
    ranges = [(int(item["start_idx"]), int(item["end_idx"])) for item in boundaries.values()]
    return sorted(ranges)


def filter_history_within_source(target_indices: np.ndarray, required_history: int, ranges: list[tuple[int, int]]) -> np.ndarray:
    if required_history <= 0 or not ranges:
        return target_indices[target_indices >= required_history]
    valid = np.zeros(len(target_indices), dtype=bool)
    for start_idx, end_idx in ranges:
        valid |= (target_indices >= start_idx + required_history) & (target_indices <= end_idx)
    return target_indices[valid]


def make_predictions(
    baseline_name: str,
    targets_raw: np.ndarray,
    target_indices: np.ndarray,
) -> np.ndarray:
    if baseline_name == "zero":
        shape = (len(target_indices), targets_raw.shape[1], targets_raw.shape[2])
        return np.zeros(shape, dtype=np.float32)
    if baseline_name == "previous_window":
        return np.asarray(targets_raw[target_indices - 1, :, :], dtype=np.float32)
    if baseline_name == "same_time_yesterday":
        return np.asarray(targets_raw[target_indices - 48, :, :], dtype=np.float32)
    if baseline_name == "same_time_last_week":
        return np.asarray(targets_raw[target_indices - 336, :, :], dtype=np.float32)
    if baseline_name == "recent_mean_4h":
        return rolling_history_mean(targets_raw, target_indices, window=8)
    if baseline_name == "recent_mean_24h":
        return rolling_history_mean(targets_raw, target_indices, window=48)
    raise ValueError(f"Unknown baseline: {baseline_name}")


def rolling_history_mean(
    targets_raw: np.ndarray,
    target_indices: np.ndarray,
    window: int,
) -> np.ndarray:
    predictions = np.empty(
        (len(target_indices), targets_raw.shape[1], targets_raw.shape[2]),
        dtype=np.float32,
    )
    for row, target_idx in enumerate(target_indices):
        predictions[row] = np.mean(
            targets_raw[target_idx - window : target_idx, :, :],
            axis=0,
            dtype=np.float64,
        )
    return predictions


def init_accumulators() -> dict[str, float | int]:
    return {
        "total_abs": 0.0,
        "total_sq": 0.0,
        "total_count": 0,
        "rental_abs": 0.0,
        "rental_sq": 0.0,
        "rental_count": 0,
        "return_abs": 0.0,
        "return_sq": 0.0,
        "return_count": 0,
        "net_abs": 0.0,
        "net_sq": 0.0,
        "net_count": 0,
        "skipped_timestamps": 0,
        "evaluated_timestamps": 0,
    }


def update_accumulators(acc: dict[str, float | int], pred: np.ndarray, true: np.ndarray) -> None:
    diff = pred.astype(np.float64) - true.astype(np.float64)
    abs_diff = np.abs(diff)
    sq_diff = np.square(diff)

    chunk_t, num_stations, num_targets = diff.shape
    acc["total_abs"] += float(abs_diff.sum())
    acc["total_sq"] += float(sq_diff.sum())
    acc["total_count"] += int(chunk_t * num_stations * num_targets)

    rental_diff = diff[:, :, 0]
    return_diff = diff[:, :, 1]
    pred_net = pred[:, :, 0].astype(np.float64) - pred[:, :, 1].astype(np.float64)
    true_net = true[:, :, 0].astype(np.float64) - true[:, :, 1].astype(np.float64)
    net_diff = pred_net - true_net
    acc["rental_abs"] += float(np.abs(rental_diff).sum())
    acc["rental_sq"] += float(np.square(rental_diff).sum())
    acc["rental_count"] += int(chunk_t * num_stations)
    acc["return_abs"] += float(np.abs(return_diff).sum())
    acc["return_sq"] += float(np.square(return_diff).sum())
    acc["return_count"] += int(chunk_t * num_stations)
    acc["net_abs"] += float(np.abs(net_diff).sum())
    acc["net_sq"] += float(np.square(net_diff).sum())
    acc["net_count"] += int(chunk_t * num_stations)
    acc["evaluated_timestamps"] += int(chunk_t)


def finalize_metrics(acc: dict[str, float | int]) -> dict[str, float | int]:
    if acc["total_count"] == 0:
        return {
            "total_mae": None,
            "total_rmse": None,
            "rental_mae": None,
            "rental_rmse": None,
            "return_mae": None,
            "return_rmse": None,
            "net_demand_mae": None,
            "net_demand_rmse": None,
            "evaluated_timestamps": 0,
            "skipped_timestamps": int(acc["skipped_timestamps"]),
        }
    return {
        "total_mae": float(acc["total_abs"] / acc["total_count"]),
        "total_rmse": float(np.sqrt(acc["total_sq"] / acc["total_count"])),
        "rental_mae": float(acc["rental_abs"] / acc["rental_count"]),
        "rental_rmse": float(np.sqrt(acc["rental_sq"] / acc["rental_count"])),
        "return_mae": float(acc["return_abs"] / acc["return_count"]),
        "return_rmse": float(np.sqrt(acc["return_sq"] / acc["return_count"])),
        "net_demand_mae": float(acc["net_abs"] / acc["net_count"]),
        "net_demand_rmse": float(np.sqrt(acc["net_sq"] / acc["net_count"])),
        "evaluated_timestamps": int(acc["evaluated_timestamps"]),
        "skipped_timestamps": int(acc["skipped_timestamps"]),
    }


def compute_metrics_for_baseline(
    baseline_name: str,
    targets_raw: np.ndarray,
    split_indices: np.ndarray,
    source_ranges: list[tuple[int, int]],
    chunk_size: int,
) -> dict[str, float | int]:
    required_history = int(BASELINES[baseline_name]["history"])
    valid_indices = filter_history_within_source(split_indices, required_history, source_ranges)
    acc = init_accumulators()
    acc["skipped_timestamps"] = int(len(split_indices) - len(valid_indices))

    for start in range(0, len(valid_indices), chunk_size):
        chunk_indices = valid_indices[start : start + chunk_size]
        true = np.asarray(targets_raw[chunk_indices, :, :], dtype=np.float32)
        pred = make_predictions(baseline_name, targets_raw, chunk_indices)
        update_accumulators(acc, pred, true)

    return finalize_metrics(acc)


def evaluate_split(
    split_name: str,
    split_indices: np.ndarray,
    targets_raw: np.ndarray,
    source_ranges: list[tuple[int, int]],
    chunk_size: int,
) -> dict[str, dict[str, float | int]]:
    print(f"Evaluating split: {split_name} ({len(split_indices)} target timestamps)")
    return {
        baseline_name: compute_metrics_for_baseline(
            baseline_name,
            targets_raw,
            split_indices,
            source_ranges,
            chunk_size,
        )
        for baseline_name in BASELINES
    }


def flatten_baseline_for_wandb(
    results: dict[str, dict[str, dict[str, float | int | None]]],
    baseline_name: str,
) -> dict[str, float | int]:
    payload: dict[str, float | int] = {}
    for split_name, split_results in results.items():
        metrics = split_results[baseline_name]
        for metric_name, value in metrics.items():
            if value is not None:
                payload[f"{split_name}/{metric_name}"] = value
    return payload


def ensure_wandb_login() -> None:
    if "WANDB_API_KEY" in os.environ:
        return
    try:
        from google.colab import userdata  # type: ignore

        wandb_key = userdata.get("WANDB_API_KEY")
        if wandb_key:
            os.environ["WANDB_API_KEY"] = wandb_key
    except Exception:
        pass


def init_wandb(args: argparse.Namespace, array_dir: Path, baseline_name: str):
    if not args.wandb_enabled:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb is enabled but not installed. Run `pip install wandb`.") from exc

    ensure_wandb_login()

    data_version = args.data_dir.name
    run_name = f"{args.wandb_name}_{baseline_name}" if args.wandb_name else baseline_name
    group = args.wandb_group or data_version
    return wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=run_name,
        group=group,
        job_type=args.wandb_job_type,
        config={
            "data_dir": str(args.data_dir),
            "array_dir": str(array_dir),
            "splits": args.splits,
            "chunk_size": args.chunk_size,
            "baseline": baseline_name,
            "baseline_config": BASELINES[baseline_name],
        },
    )


def log_results_to_wandb(
    args: argparse.Namespace,
    array_dir: Path,
    results: dict[str, dict[str, dict[str, float | int | None]]],
) -> None:
    if not args.wandb_enabled:
        return
    for baseline_name in BASELINES:
        wandb_run = init_wandb(args, array_dir, baseline_name)
        payload = flatten_baseline_for_wandb(results, baseline_name)
        wandb_run.log(payload)
        wandb_run.summary.update(payload)
        wandb_run.finish()


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_csv(path: Path, results: dict[str, dict[str, dict[str, float | int]]]) -> None:
    rows = []
    for split_name, split_results in results.items():
        for baseline_name, metrics in split_results.items():
            rows.append(
                {
                    "split": split_name,
                    "baseline": baseline_name,
                    **metrics,
                }
            )
    fieldnames = [
        "split",
        "baseline",
        "total_mae",
        "total_rmse",
        "rental_mae",
        "rental_rmse",
        "return_mae",
        "return_rmse",
        "net_demand_mae",
        "net_demand_rmse",
        "evaluated_timestamps",
        "skipped_timestamps",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_metric(value: float | int | None) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def print_results_table(results: dict[str, dict[str, dict[str, float | int]]]) -> None:
    headers = [
        "split",
        "baseline",
        "total_mae",
        "total_rmse",
        "rental_mae",
        "rental_rmse",
        "return_mae",
        "return_rmse",
        "net_demand_mae",
        "net_demand_rmse",
        "eval_T",
        "skip_T",
    ]
    rows = []
    for split_name, split_results in results.items():
        for baseline_name, metrics in split_results.items():
            rows.append(
                [
                    split_name,
                    baseline_name,
                    format_metric(metrics["total_mae"]),
                    format_metric(metrics["total_rmse"]),
                    format_metric(metrics["rental_mae"]),
                    format_metric(metrics["rental_rmse"]),
                    format_metric(metrics["return_mae"]),
                    format_metric(metrics["return_rmse"]),
                    format_metric(metrics["net_demand_mae"]),
                    format_metric(metrics["net_demand_rmse"]),
                    format_metric(metrics["evaluated_timestamps"]),
                    format_metric(metrics["skipped_timestamps"]),
                ]
            )

    widths = [
        max(len(str(row[col_idx])) for row in [headers] + rows)
        for col_idx in range(len(headers))
    ]
    print()
    print("Naive baseline metrics")
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    print(" | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(headers)))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    for row in rows:
        print(" | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(row)))
    print()


def main() -> None:
    args = parse_args()
    array_dir = resolve_array_dir(args.data_dir)
    targets_path = array_dir / "targets_raw.npy"
    timestamps_path = array_dir / "timestamps.npy"
    splits_path = args.data_dir / "splits.json"

    missing = [path for path in (targets_path, timestamps_path, splits_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {[str(path) for path in missing]}")

    targets_raw = np.load(targets_path, mmap_mode="r")
    _timestamps = np.load(timestamps_path, mmap_mode="r")
    split_indices = load_split_indices(args.data_dir)
    source_ranges = load_source_ranges(args.data_dir)

    if targets_raw.ndim != 3 or targets_raw.shape[2] != 2:
        raise ValueError(f"Expected targets_raw shape (T, S, 2), got {targets_raw.shape}")

    results = {}
    for split_name in args.splits:
        if split_name not in split_indices:
            raise ValueError(f"Unknown split '{split_name}'. Available: {sorted(split_indices)}")
        results[split_name] = evaluate_split(
            split_name,
            split_indices[split_name],
            targets_raw,
            source_ranges,
            args.chunk_size,
        )

    print_results_table(results)
    log_results_to_wandb(args, array_dir, results)

    if args.save_local:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        save_json(args.output_dir / "naive_baseline_metrics.json", results)
        save_csv(args.output_dir / "naive_baseline_metrics.csv", results)
        print(f"Saved JSON metrics to {args.output_dir / 'naive_baseline_metrics.json'}")
        print(f"Saved CSV metrics to {args.output_dir / 'naive_baseline_metrics.csv'}")


if __name__ == "__main__":
    main()
