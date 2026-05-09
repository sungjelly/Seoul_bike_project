import torch

from src.models.tct_gat.temporal_encoder import TriScaleTemporalEncoder


def test_triscale_temporal_encoder_forward_shape():
    encoder = TriScaleTemporalEncoder(
        input_dim=5,
        window_offsets=[-4, -3, -2, -1],
        recent_offsets=[-2, -1],
        daily_offsets=[-3],
        weekly_offsets=[-4],
        recent_hidden_dim=4,
        daily_hidden_dim=3,
        weekly_hidden_dim=2,
    )
    x = torch.randn(1, 3, 4, 5)
    out = encoder(x)
    assert out.shape == (1, 3, 9)
