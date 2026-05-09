from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


TARGET_COLUMNS = ["rental_count", "return_count"]


def load_target_scalers(data_dir: str | Path) -> dict:
    with (Path(data_dir) / "scalers.json").open("r", encoding="utf-8") as f:
        scalers = json.load(f)
    return scalers["target"]


def inverse_transform_targets(values_scaled: torch.Tensor | np.ndarray, target_scalers: dict) -> np.ndarray:
    values = values_scaled.detach().cpu().numpy() if isinstance(values_scaled, torch.Tensor) else np.asarray(values_scaled)
    out = np.empty_like(values, dtype=np.float64)
    for idx, column in enumerate(TARGET_COLUMNS):
        scaler = target_scalers[column]
        if scaler.get("transform") != "log1p":
            raise ValueError(f"Unsupported target transform for {column}: {scaler.get('transform')}")
        unscaled_log = values[..., idx] * float(scaler["std"]) + float(scaler["mean"])
        out[..., idx] = np.expm1(unscaled_log)
    return np.clip(out, 0.0, np.inf)


class RawCountMetricAccumulator:
    """Accumulates one-step or rollout raw-count metrics."""

    def __init__(self, prefix: str = "") -> None:
        self.prefix = prefix
        self.count = 0
        self.rental_abs = 0.0
        self.rental_sq = 0.0
        self.return_abs = 0.0
        self.return_sq = 0.0
        self.horizon_rental_abs: np.ndarray | None = None
        self.horizon_return_abs: np.ndarray | None = None
        self.horizon_count: np.ndarray | None = None

    def update(self, pred_raw: torch.Tensor | np.ndarray, target_raw: torch.Tensor | np.ndarray) -> None:
        pred = pred_raw.detach().cpu().numpy() if isinstance(pred_raw, torch.Tensor) else np.asarray(pred_raw)
        true = target_raw.detach().cpu().numpy() if isinstance(target_raw, torch.Tensor) else np.asarray(target_raw)
        pred = np.asarray(pred, dtype=np.float64)
        true = np.asarray(true, dtype=np.float64)
        if pred.shape != true.shape:
            raise ValueError(f"Prediction and target shapes differ: {pred.shape} vs {true.shape}")
        if pred.ndim == 3:
            pred = pred[:, None, :, :]
            true = true[:, None, :, :]
        if pred.ndim != 4 or pred.shape[-1] != 2:
            raise ValueError(f"Metrics expect B x S x 2 or B x H x S x 2, got {pred.shape}")

        diff = pred - true
        rental = diff[..., 0].reshape(-1)
        returns = diff[..., 1].reshape(-1)
        self.count += int(rental.size)
        self.rental_abs += float(np.abs(rental).sum())
        self.rental_sq += float(np.square(rental).sum())
        self.return_abs += float(np.abs(returns).sum())
        self.return_sq += float(np.square(returns).sum())

        horizon_count = pred.shape[0] * pred.shape[2]
        rental_abs = np.abs(diff[..., 0]).sum(axis=(0, 2))
        return_abs = np.abs(diff[..., 1]).sum(axis=(0, 2))
        if self.horizon_rental_abs is None:
            self.horizon_rental_abs = np.zeros(pred.shape[1], dtype=np.float64)
            self.horizon_return_abs = np.zeros(pred.shape[1], dtype=np.float64)
            self.horizon_count = np.zeros(pred.shape[1], dtype=np.int64)
        self.horizon_rental_abs += rental_abs
        self.horizon_return_abs += return_abs
        self.horizon_count += horizon_count

    def compute(self) -> dict[str, float]:
        if self.count == 0:
            raise ValueError("No samples accumulated for metrics.")
        rental_mae = self.rental_abs / self.count
        return_mae = self.return_abs / self.count
        rental_rmse = float(np.sqrt(self.rental_sq / self.count))
        return_rmse = float(np.sqrt(self.return_sq / self.count))
        metrics = {
            "total_mae": float((rental_mae + return_mae) / 2.0),
            "total_rmse": float((rental_rmse + return_rmse) / 2.0),
            "rental_mae": float(rental_mae),
            "rental_rmse": float(rental_rmse),
            "return_mae": float(return_mae),
            "return_rmse": float(return_rmse),
        }
        if self.prefix:
            metrics = {f"{self.prefix}_{key}": value for key, value in metrics.items()}
        if self.horizon_rental_abs is not None and self.horizon_return_abs is not None and self.horizon_count is not None:
            for idx in range(len(self.horizon_count)):
                denom = max(int(self.horizon_count[idx]), 1)
                rental_h = float(self.horizon_rental_abs[idx] / denom)
                return_h = float(self.horizon_return_abs[idx] / denom)
                if len(self.horizon_count) > 1:
                    metrics[f"horizon_{idx + 1}_total_mae"] = float((rental_h + return_h) / 2.0)
                    metrics[f"horizon_{idx + 1}_rental_mae"] = rental_h
                    metrics[f"horizon_{idx + 1}_return_mae"] = return_h
        return metrics
