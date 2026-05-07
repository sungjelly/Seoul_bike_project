from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


TARGET_COLUMNS = ["rental_count", "return_count"]


def load_target_scalers(data_dir: str | Path) -> dict:
    scalers_path = Path(data_dir) / "scalers.json"
    with scalers_path.open("r", encoding="utf-8") as f:
        scalers = json.load(f)
    if "count_scaler" in scalers:
        count_scaler = scalers["count_scaler"]
        return {
            "rental_count": count_scaler,
            "return_count": count_scaler,
        }
    return scalers["target"]


def inverse_transform_targets(values_scaled: torch.Tensor | np.ndarray, target_scalers: dict) -> np.ndarray:
    if isinstance(values_scaled, torch.Tensor):
        values = values_scaled.detach().cpu().numpy()
    else:
        values = np.asarray(values_scaled)

    output = np.empty_like(values, dtype=np.float64)
    for idx, column in enumerate(TARGET_COLUMNS):
        scaler = target_scalers[column]
        if scaler["transform"] != "log1p":
            raise ValueError(f"Unsupported target transform for {column}: {scaler['transform']}")
        pred_log = values[..., idx] * float(scaler["std"]) + float(scaler["mean"])
        output[..., idx] = np.expm1(pred_log)
    return np.clip(output, 0.0, np.inf)


class RawCountMetricAccumulator:
    """Accumulates raw-count metrics for multi-horizon lstm2 predictions."""

    def __init__(self) -> None:
        self.count = 0
        self.rental_abs = 0.0
        self.rental_sq = 0.0
        self.return_abs = 0.0
        self.return_sq = 0.0
        self.net_abs = 0.0
        self.net_sq = 0.0
        self.horizon_total_abs: np.ndarray | None = None
        self.horizon_count: np.ndarray | None = None

    def update(self, pred_raw: np.ndarray, target_raw: torch.Tensor | np.ndarray) -> None:
        true = target_raw.detach().cpu().numpy() if isinstance(target_raw, torch.Tensor) else np.asarray(target_raw)
        pred = np.asarray(pred_raw, dtype=np.float64)
        true = np.asarray(true, dtype=np.float64)
        if pred.shape != true.shape:
            raise ValueError(f"Prediction and target shapes differ: {pred.shape} vs {true.shape}")
        if pred.ndim != 3 or pred.shape[-1] != 2:
            raise ValueError(f"Expected raw metrics inputs with shape (B, horizon, 2), got {pred.shape}")

        diff = pred - true
        rental_diff = diff[..., 0].reshape(-1)
        return_diff = diff[..., 1].reshape(-1)
        pred_net = pred[..., 0] - pred[..., 1]
        true_net = true[..., 0] - true[..., 1]
        net_diff = (pred_net - true_net).reshape(-1)

        batch_count = int(rental_diff.size)
        self.count += batch_count
        self.rental_abs += float(np.abs(rental_diff).sum())
        self.rental_sq += float(np.square(rental_diff).sum())
        self.return_abs += float(np.abs(return_diff).sum())
        self.return_sq += float(np.square(return_diff).sum())
        self.net_abs += float(np.abs(net_diff).sum())
        self.net_sq += float(np.square(net_diff).sum())

        total_abs_by_horizon = (np.abs(diff[..., 0]) + np.abs(diff[..., 1])) / 2.0
        horizon_abs = total_abs_by_horizon.sum(axis=0)
        horizon_count = np.full(pred.shape[1], pred.shape[0], dtype=np.int64)
        if self.horizon_total_abs is None:
            self.horizon_total_abs = np.zeros(pred.shape[1], dtype=np.float64)
            self.horizon_count = np.zeros(pred.shape[1], dtype=np.int64)
        self.horizon_total_abs += horizon_abs
        self.horizon_count += horizon_count

    def compute(self) -> dict[str, float]:
        if self.count == 0:
            raise ValueError("No samples were accumulated for metrics.")
        rental_mae = self.rental_abs / self.count
        rental_rmse = float(np.sqrt(self.rental_sq / self.count))
        return_mae = self.return_abs / self.count
        return_rmse = float(np.sqrt(self.return_sq / self.count))
        net_mae = self.net_abs / self.count
        net_rmse = float(np.sqrt(self.net_sq / self.count))
        metrics = {
            "rental_mae": float(rental_mae),
            "rental_rmse": float(rental_rmse),
            "return_mae": float(return_mae),
            "return_rmse": float(return_rmse),
            "net_demand_mae": float(net_mae),
            "net_demand_rmse": float(net_rmse),
            "total_mae": float((rental_mae + return_mae) / 2.0),
            "total_rmse": float((rental_rmse + return_rmse) / 2.0),
        }
        if self.horizon_total_abs is not None and self.horizon_count is not None:
            for idx, value in enumerate(self.horizon_total_abs / np.maximum(self.horizon_count, 1), start=1):
                metrics[f"h{idx}_total_mae"] = float(value)
        return metrics
