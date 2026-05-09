from __future__ import annotations

import torch


def _normal(shape: torch.Size, ref: torch.Tensor, std: float, generator: torch.Generator | None) -> torch.Tensor:
    if std <= 0:
        return torch.zeros(shape, dtype=ref.dtype, device=ref.device)
    return torch.randn(shape, dtype=ref.dtype, device=ref.device, generator=generator) * std


def apply_future_weather_noise(
    future_weather_features: torch.Tensor,
    config: dict | None,
    enabled: bool,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply graph-level forecast uncertainty in scaled feature space.

    The input shape is B x 4 for one-step or B x H x 4 for rollout. The four
    columns are temperature, wind_speed, rainfall, humidity. This initial
    implementation operates in scaled space, so config std values are interpreted
    as scaled-feature perturbation magnitudes.
    """

    if not enabled:
        return future_weather_features
    if future_weather_features.ndim not in {2, 3} or future_weather_features.shape[-1] != 4:
        raise ValueError(f"future_weather_features must have shape B x 4 or B x H x 4, got {tuple(future_weather_features.shape)}")
    cfg = config or {}
    out = future_weather_features.clone()
    out[..., 0] = out[..., 0] + _normal(out[..., 0].shape, out, float(cfg.get("temperature_std", 0.0)), generator)
    out[..., 1] = out[..., 1] + _normal(out[..., 1].shape, out, float(cfg.get("wind_speed_std", 0.0)), generator)
    out[..., 3] = out[..., 3] + _normal(out[..., 3].shape, out, float(cfg.get("humidity_std", 0.0)), generator)

    rainfall = out[..., 2]
    scale_std = float(cfg.get("rainfall_intensity_scale_std", 0.0))
    flip_prob = float(cfg.get("rainfall_occurrence_flip_prob", 0.0))
    if scale_std > 0:
        rainfall = rainfall + _normal(rainfall.shape, out, scale_std, generator)
    if flip_prob > 0:
        flips = torch.rand(rainfall.shape, dtype=out.dtype, device=out.device, generator=generator) < flip_prob
        rainfall = torch.where(flips, -rainfall, rainfall)
    out[..., 2] = rainfall
    return out
