from __future__ import annotations

import torch

from src.training.tct_gat_training.weather_scenarios import apply_future_weather_noise


def _gather_window(history: torch.Tensor, window_offsets: torch.Tensor) -> torch.Tensor:
    if history.ndim != 4:
        raise ValueError(f"history must have shape B x history x S x F, got {tuple(history.shape)}.")
    offsets = window_offsets.to(device=history.device, dtype=torch.long)
    if torch.any(offsets >= 0):
        raise ValueError("window_offsets must be negative.")
    indices = history.shape[1] + offsets
    if int(indices.min()) < 0 or int(indices.max()) >= history.shape[1]:
        raise ValueError("History buffer is too short for selected window offsets.")
    return history.index_select(1, indices).transpose(1, 2).contiguous()


@torch.no_grad()
def rollout_autoregressive(
    model: torch.nn.Module,
    initial_rental_history: torch.Tensor,
    initial_return_history: torch.Tensor,
    future_target_time_features: torch.Tensor,
    future_weather_features: torch.Tensor,
    static_numeric: torch.Tensor,
    station_index: torch.Tensor,
    district_id: torch.Tensor,
    operation_type_id: torch.Tensor | None,
    window_offsets: torch.Tensor,
    scalers: dict,
    n_horizons: int,
    weather_noise_config: dict | None = None,
    apply_weather_noise_eval: bool = False,
) -> torch.Tensor:
    """Roll out one-step TCT-GAT predictions.

    Returns scaled predictions with shape B x H x S x 2. Histories are
    contiguous scaled feature buffers from min(window_offsets) through t-1 with
    shapes B x history x S x 5.
    """

    if n_horizons <= 0:
        raise ValueError(f"n_horizons must be positive, got {n_horizons}.")
    if initial_rental_history.shape != initial_return_history.shape:
        raise ValueError("Initial rental and return histories must have matching shape.")
    if initial_rental_history.ndim != 4 or initial_rental_history.shape[-1] != 5:
        raise ValueError("Initial histories must have shape B x history x S x 5.")
    if future_target_time_features.ndim != 3 or future_target_time_features.shape[-1] != 8:
        raise ValueError("future_target_time_features must have shape B x H x 8.")
    if future_weather_features.ndim != 3 or future_weather_features.shape[-1] != 4:
        raise ValueError("future_weather_features must have shape B x H x 4.")

    rental_history = initial_rental_history
    return_history = initial_return_history
    predictions: list[torch.Tensor] = []
    for horizon_idx in range(n_horizons):
        rental_seq = _gather_window(rental_history, window_offsets)
        return_seq = _gather_window(return_history, window_offsets)
        target_time = future_target_time_features[:, horizon_idx, :]
        weather = future_weather_features[:, horizon_idx, :]
        weather = apply_future_weather_noise(weather, weather_noise_config, enabled=apply_weather_noise_eval)
        pred_scaled = model(
            rental_seq=rental_seq,
            return_seq=return_seq,
            target_time_features=target_time,
            future_weather_features=weather,
            static_numeric=static_numeric,
            station_index=station_index,
            district_id=district_id,
            operation_type_id=operation_type_id,
        )
        if pred_scaled.ndim != 3 or pred_scaled.shape[-1] != 2:
            raise ValueError(f"TCT-GAT one-step output must have shape B x S x 2, got {tuple(pred_scaled.shape)}.")
        predictions.append(pred_scaled)
        weather_station = weather[:, None, :].expand(-1, pred_scaled.shape[1], -1)
        next_rental = torch.cat([pred_scaled[..., 0:1], weather_station], dim=-1).unsqueeze(1)
        next_return = torch.cat([pred_scaled[..., 1:2], weather_station], dim=-1).unsqueeze(1)
        rental_history = torch.cat([rental_history, next_rental], dim=1)
        return_history = torch.cat([return_history, next_return], dim=1)
    return torch.stack(predictions, dim=1)
