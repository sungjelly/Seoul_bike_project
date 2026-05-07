from __future__ import annotations

import torch
from torch import nn


class BaselineLSTM(nn.Module):
    """Station-level LSTM baseline for next-window demand prediction.

    The model outputs transformed/scaled targets. Raw-count inverse transforms
    are intentionally kept outside the model for evaluation and inference.
    """

    def __init__(
        self,
        input_dim: int,
        static_numeric_dim: int,
        output_dim: int,
        num_stations: int,
        num_districts: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        station_embedding_dim: int = 16,
        district_embedding_dim: int = 8,
        mlp_hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.station_embedding = nn.Embedding(num_stations, station_embedding_dim)
        self.district_embedding = nn.Embedding(num_districts, district_embedding_dim)

        concat_dim = (
            hidden_dim
            + station_embedding_dim
            + district_embedding_dim
            + static_numeric_dim
        )
        second_hidden = max(mlp_hidden_dim // 2, 1)
        self.head = nn.Sequential(
            nn.Linear(concat_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, second_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(second_hidden, output_dim),
        )

    def forward(
        self,
        x_seq: torch.Tensor,
        static_numeric: torch.Tensor,
        station_index: torch.Tensor,
        district_id: torch.Tensor,
    ) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x_seq)
        final_hidden = hidden[-1]
        station_emb = self.station_embedding(station_index)
        district_emb = self.district_embedding(district_id)
        features = torch.cat([final_hidden, station_emb, district_emb, static_numeric], dim=1)
        return self.head(features)
