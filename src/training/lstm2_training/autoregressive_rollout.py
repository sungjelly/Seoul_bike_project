from __future__ import annotations

import torch

from src.training.lstm2_training.weather_uncertainty import apply_weather_noise


def signed_log1p(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def _require_scaler(scalers: dict, key: str) -> dict:
    if key not in scalers:
        raise ValueError(f"Missing scaler key {key!r}; cannot build autoregressive dynamic history.")
    scaler = scalers[key]
    if not isinstance(scaler, dict) or "mean" not in scaler or "std" not in scaler:
        raise ValueError(f"Scaler {key!r} must contain mean and std.")
    return scaler


def _inverse_scaled_counts(pred_scaled: torch.Tensor, scalers: dict) -> tuple[torch.Tensor, torch.Tensor]:
    count_scaler = _require_scaler(scalers, "count_scaler")
    if count_scaler.get("transform") != "log1p":
        raise ValueError(f"count_scaler transform must be log1p, got {count_scaler.get('transform')}.")
    mean = torch.as_tensor(float(count_scaler["mean"]), dtype=pred_scaled.dtype, device=pred_scaled.device)
    std = torch.as_tensor(float(count_scaler["std"]), dtype=pred_scaled.dtype, device=pred_scaled.device)
    raw = torch.expm1(pred_scaled * std + mean).clamp_min(0.0)
    return raw[:, 0], raw[:, 1]


def _scaled_net_demand(raw_rental: torch.Tensor, raw_return: torch.Tensor, scalers: dict) -> torch.Tensor:
    net_scaler = _require_scaler(scalers, "net_demand_scaler")
    if net_scaler.get("transform") != "signed_log1p":
        raise ValueError(f"net_demand_scaler transform must be signed_log1p, got {net_scaler.get('transform')}.")
    mean = torch.as_tensor(float(net_scaler["mean"]), dtype=raw_rental.dtype, device=raw_rental.device)
    std = torch.as_tensor(float(net_scaler["std"]), dtype=raw_rental.dtype, device=raw_rental.device)
    raw_net = raw_rental - raw_return
    return (signed_log1p(raw_net) - mean) / std


def _build_x_seq(dynamic_history: torch.Tensor, window_offsets: torch.Tensor) -> torch.Tensor:
    if dynamic_history.ndim != 3 or dynamic_history.shape[-1] != 7:
        raise ValueError(f"initial_dynamic_history must have shape (B, history, 7), got {tuple(dynamic_history.shape)}.")
    if window_offsets.ndim != 1 or len(window_offsets) == 0:
        raise ValueError("window_offsets must be a non-empty 1D tensor.")
    if torch.any(window_offsets >= 0):
        raise ValueError("window_offsets must be negative for autoregressive rollout.")
    ordered_offsets = torch.sort(window_offsets.to(device=dynamic_history.device, dtype=torch.long)).values
    indices = dynamic_history.shape[1] + ordered_offsets
    if int(indices.min()) < 0 or int(indices.max()) >= dynamic_history.shape[1]:
        raise ValueError(
            "initial_dynamic_history is too short for the requested window_offsets: "
            f"history={dynamic_history.shape[1]}, offsets={ordered_offsets.detach().cpu().tolist()}."
        )
    return dynamic_history.index_select(1, indices)


@torch.no_grad()
def rollout_autoregressive(
    model: torch.nn.Module,
    initial_dynamic_history: torch.Tensor,
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
    """Roll out a one-step TTSLSTM2V2 model over multiple horizons.

    Returns scaled predictions with shape (B, n_horizons, 2). Dynamic feature
    order is rental_count, return_count, net_demand, temperature, wind_speed,
    rainfall, humidity.
    """

    if n_horizons <= 0:
        raise ValueError(f"n_horizons must be positive, got {n_horizons}.")
    if future_target_time_features.ndim != 3 or future_target_time_features.shape[-1] != 8:
        raise ValueError(
            "future_target_time_features must have shape (B, n_horizons, 8), "
            f"got {tuple(future_target_time_features.shape)}."
        )
    if future_weather_features.ndim != 3 or future_weather_features.shape[-1] != 4:
        raise ValueError(
            "future_weather_features must have shape (B, n_horizons, 4), "
            f"got {tuple(future_weather_features.shape)}."
        )
    if future_target_time_features.shape[1] < n_horizons or future_weather_features.shape[1] < n_horizons:
        raise ValueError("Future feature tensors must contain at least n_horizons steps.")
    if future_target_time_features.shape[0] != initial_dynamic_history.shape[0]:
        raise ValueError("Future target-time batch size must match initial_dynamic_history.")
    if future_weather_features.shape[0] != initial_dynamic_history.shape[0]:
        raise ValueError("Future weather batch size must match initial_dynamic_history.")

    device = initial_dynamic_history.device
    dtype = initial_dynamic_history.dtype
    history = initial_dynamic_history
    offsets = window_offsets.to(device=device, dtype=torch.long)
    static_numeric = static_numeric.to(device=device, dtype=dtype)
    station_index = station_index.to(device=device, dtype=torch.long)
    district_id = district_id.to(device=device, dtype=torch.long)
    if operation_type_id is not None:
        operation_type_id = operation_type_id.to(device=device, dtype=torch.long)
    predictions: list[torch.Tensor] = []

    for horizon_idx in range(n_horizons):
        x_seq = _build_x_seq(history, offsets)
        target_time = future_target_time_features[:, horizon_idx, :].to(device=device, dtype=dtype)
        weather = future_weather_features[:, horizon_idx, :].to(device=device, dtype=dtype)
        if weather_noise_config is not None:
            weather = apply_weather_noise(weather, weather_noise_config, enabled=apply_weather_noise_eval)

        pred_scaled = model(
            x_seq=x_seq,
            target_time_features=target_time,
            future_weather_features=weather,
            static_numeric=static_numeric,
            station_index=station_index,
            district_id=district_id,
            operation_type_id=operation_type_id,
        )
        if pred_scaled.ndim != 2 or pred_scaled.shape[-1] != 2:
            raise ValueError(f"One-step model must return shape (B, 2), got {tuple(pred_scaled.shape)}.")
        predictions.append(pred_scaled)

        raw_rental, raw_return = _inverse_scaled_counts(pred_scaled, scalers)
        # Net demand has its own signed-log transform and scaler; scaled rental
        # minus scaled return is not meaningful here.
        net_scaled = _scaled_net_demand(raw_rental, raw_return, scalers)
        next_dynamic = torch.cat([pred_scaled, net_scaled.unsqueeze(1), weather], dim=1).unsqueeze(1)
        history = torch.cat([history, next_dynamic], dim=1)

    return torch.stack(predictions, dim=1)
