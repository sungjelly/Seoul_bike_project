from __future__ import annotations

import numpy as np


def log1p_transform(x):
    return np.log1p(x)


def inverse_log1p_transform(x):
    return np.expm1(x)


def signed_log1p_transform(x):
    return np.sign(x) * np.log1p(np.abs(x))


def inverse_signed_log1p_transform(x):
    return np.sign(x) * np.expm1(np.abs(x))


def standardize(x, mean, std):
    if std == 0:
        std = 1.0
    return (x - mean) / std


def inverse_standardize(x, mean, std):
    return x * std + mean


def apply_transform(x, transform_name):
    if transform_name is None or transform_name == "none":
        return x
    if transform_name == "log1p":
        return log1p_transform(x)
    if transform_name == "signed_log1p":
        return signed_log1p_transform(x)
    raise ValueError(f"Unknown transform: {transform_name}")


def inverse_transform(x, transform_name):
    if transform_name is None or transform_name == "none":
        return x
    if transform_name == "log1p":
        return inverse_log1p_transform(x)
    if transform_name == "signed_log1p":
        return inverse_signed_log1p_transform(x)
    raise ValueError(f"Unknown transform: {transform_name}")
