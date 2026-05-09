import torch

from src.training.tct_gat_training.autoregressive_rollout import _gather_window
from src.training.tct_gat_training.metrics import inverse_transform_targets


def test_gather_window_returns_station_major_sequence():
    history = torch.randn(2, 5, 3, 5)
    window = torch.tensor([-3, -1])
    out = _gather_window(history, window)
    assert out.shape == (2, 3, 2, 5)


def test_inverse_transform_clips_nonnegative():
    scalers = {
        "rental_count": {"mean": 0.0, "std": 1.0, "transform": "log1p"},
        "return_count": {"mean": 0.0, "std": 1.0, "transform": "log1p"},
    }
    raw = inverse_transform_targets(torch.tensor([[[-10.0, -10.0]]]), scalers)
    assert raw.min() >= 0.0
