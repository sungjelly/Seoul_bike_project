from __future__ import annotations

import torch


def _normal_like(weather: torch.Tensor, column: torch.Tensor, sigma: float, generator: torch.Generator | None) -> torch.Tensor:
    if sigma <= 0.0:
        return torch.zeros_like(column)
    return torch.randn(
        column.shape,
        dtype=weather.dtype,
        device=weather.device,
        generator=generator,
    ) * sigma


def apply_rainfall_noise(
    rainfall: torch.Tensor,
    config: dict,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Perturb scaled rainfall conservatively.

    Raw-space rainfall noise should use zero-inflated occurrence plus lognormal
    intensity. The current tensors are already scaled, so this approximation
    avoids simple unconstrained Gaussian noise while keeping perturbations small.
    """

    sigma = float(config.get("rainfall_lognormal_sigma", 0.0))
    flip_prob = float(config.get("rainfall_flip_prob", 0.0))
    false_positive_scale = float(config.get("rainfall_false_positive_scale", 0.0))
    if sigma <= 0.0 and flip_prob <= 0.0:
        return rainfall

    noise = torch.randn(
        rainfall.shape,
        dtype=rainfall.dtype,
        device=rainfall.device,
        generator=generator,
    ) * sigma
    nonzero_like = rainfall.abs() > 1.0e-6
    noised = rainfall + torch.where(nonzero_like, noise.clamp(min=-2.0 * sigma, max=2.0 * sigma), torch.zeros_like(noise))

    if flip_prob > 0.0 and false_positive_scale > 0.0:
        flips = torch.rand(
            rainfall.shape,
            dtype=rainfall.dtype,
            device=rainfall.device,
            generator=generator,
        ) < flip_prob
        false_positive = torch.empty_like(rainfall).exponential_(1.0 / false_positive_scale)
        noised = torch.where(~nonzero_like & flips, rainfall + false_positive, noised)
    return noised


def apply_weather_noise(
    weather: torch.Tensor,
    config: dict,
    enabled: bool,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Simulate forecast weather error during training.

    Input weather has shape (B, 4) and feature order:
    temperature, wind_speed, rainfall, humidity. The first implementation
    perturbs scaled model features; raw-space support can be added once inverse
    scaler plumbing is available here.
    """

    if not enabled:
        return weather
    if weather.ndim != 2 or weather.shape[-1] != 4:
        raise ValueError(f"weather must have shape (B, 4), got {tuple(weather.shape)}.")

    out = weather.clone()
    out[:, 0] = out[:, 0] + _normal_like(out, out[:, 0], float(config.get("temperature_sigma", 0.0)), generator)
    out[:, 1] = out[:, 1] + _normal_like(out, out[:, 1], float(config.get("wind_speed_sigma", 0.0)), generator)
    out[:, 2] = apply_rainfall_noise(out[:, 2], config, generator)
    out[:, 3] = out[:, 3] + _normal_like(out, out[:, 3], float(config.get("humidity_sigma", 0.0)), generator)
    return out
