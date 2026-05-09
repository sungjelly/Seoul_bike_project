from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn

from src.models.tct_gat.edge_aware_gat import EdgeAwareHeteroGAT
from src.models.tct_gat.station_context_encoder import StationContextEncoder
from src.models.tct_gat.temporal_encoder import TriScaleTemporalEncoder


def make_mlp(input_dim: int, output_dim: int, hidden_dims: Sequence[int], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = int(input_dim)
    for hidden in hidden_dims:
        hidden_dim = int(hidden)
        layers.extend([nn.Linear(prev, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
        prev = hidden_dim
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class TCTGAT1AR(nn.Module):
    """One-step autoregressive graph-snapshot TCT-GAT model."""

    def __init__(
        self,
        graph_dir: str | Path,
        input_dim: int,
        target_time_dim: int,
        future_weather_dim: int,
        output_dim: int,
        num_stations: int,
        num_districts: int,
        num_operation_types: int,
        static_numeric_dim: int,
        window_offsets: Sequence[int],
        recent_offsets: Sequence[int],
        daily_offsets: Sequence[int],
        weekly_offsets: Sequence[int],
        recent_hidden_dim: int = 64,
        daily_hidden_dim: int = 32,
        weekly_hidden_dim: int = 32,
        recent_num_layers: int = 1,
        daily_num_layers: int = 1,
        weekly_num_layers: int = 1,
        station_embedding_dim: int = 16,
        district_embedding_dim: int = 8,
        operation_type_embedding_dim: int = 1,
        static_context_dim: int = 32,
        token_dim: int = 128,
        gat_layers: int = 1,
        gat_heads: int = 4,
        edge_embedding_dim: int = 16,
        decoder_hidden_dims: Sequence[int] | None = None,
        dropout: float = 0.2,
        gat_dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim != 5 or target_time_dim != 8 or future_weather_dim != 4 or output_dim != 2:
            raise ValueError("TCTGAT1AR requires input_dim=5, target_time_dim=8, future_weather_dim=4, output_dim=2.")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.rental_temporal = TriScaleTemporalEncoder(
            input_dim,
            window_offsets,
            recent_offsets,
            daily_offsets,
            weekly_offsets,
            recent_hidden_dim,
            daily_hidden_dim,
            weekly_hidden_dim,
            recent_num_layers,
            daily_num_layers,
            weekly_num_layers,
            dropout,
        )
        self.return_temporal = TriScaleTemporalEncoder(
            input_dim,
            window_offsets,
            recent_offsets,
            daily_offsets,
            weekly_offsets,
            recent_hidden_dim,
            daily_hidden_dim,
            weekly_hidden_dim,
            recent_num_layers,
            daily_num_layers,
            weekly_num_layers,
            dropout,
        )
        self.station_context = StationContextEncoder(
            num_stations=num_stations,
            num_districts=num_districts,
            num_operation_types=num_operation_types,
            static_numeric_dim=static_numeric_dim,
            station_embedding_dim=station_embedding_dim,
            district_embedding_dim=district_embedding_dim,
            operation_type_embedding_dim=operation_type_embedding_dim,
            static_context_dim=static_context_dim,
            dropout=dropout,
        )
        temporal_dim = recent_hidden_dim + daily_hidden_dim + weekly_hidden_dim
        self.rental_token = make_mlp(temporal_dim + static_context_dim, token_dim, [token_dim], dropout)
        self.return_token = make_mlp(temporal_dim + static_context_dim, token_dim, [token_dim], dropout)
        self.gat = EdgeAwareHeteroGAT(
            graph_dir=graph_dir,
            token_dim=token_dim,
            layers=gat_layers,
            heads=gat_heads,
            edge_embedding_dim=edge_embedding_dim,
            dropout=gat_dropout,
            attention_dropout=attention_dropout,
        )
        decoder_input = 2 * token_dim + static_context_dim + target_time_dim + future_weather_dim
        hidden_dims = list(decoder_hidden_dims or [128, 64])
        self.rental_head = make_mlp(decoder_input, 1, hidden_dims, dropout)
        self.return_head = make_mlp(decoder_input, 1, hidden_dims, dropout)

    def forward(
        self,
        rental_seq: torch.Tensor,
        return_seq: torch.Tensor,
        target_time_features: torch.Tensor,
        future_weather_features: torch.Tensor,
        static_numeric: torch.Tensor,
        station_index: torch.Tensor,
        district_id: torch.Tensor,
        operation_type_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if rental_seq.ndim != 4 or return_seq.ndim != 4:
            raise ValueError("rental_seq and return_seq must have shape B x S x W x 5.")
        if rental_seq.shape != return_seq.shape:
            raise ValueError(f"rental_seq and return_seq shapes differ: {tuple(rental_seq.shape)} vs {tuple(return_seq.shape)}")
        if rental_seq.shape[-1] != self.input_dim:
            raise ValueError(f"TCT-GAT input feature dim must be {self.input_dim}.")
        batch_size, num_stations, _, _ = rental_seq.shape
        if target_time_features.shape != (batch_size, 8):
            raise ValueError(f"target_time_features must have shape B x 8, got {tuple(target_time_features.shape)}.")
        if future_weather_features.shape != (batch_size, 4):
            raise ValueError(f"future_weather_features must have shape B x 4, got {tuple(future_weather_features.shape)}.")

        station_context = self.station_context(
            static_numeric=static_numeric,
            station_index=station_index,
            district_id=district_id,
            operation_type_id=operation_type_id,
            batch_size=batch_size,
        )
        rental_time = self.rental_temporal(rental_seq)
        return_time = self.return_temporal(return_seq)
        rental_token = self.rental_token(torch.cat([rental_time, station_context], dim=-1))
        return_token = self.return_token(torch.cat([return_time, station_context], dim=-1))
        rental_final, return_final = self.gat(rental_token, return_token)

        target_context = torch.cat([target_time_features, future_weather_features], dim=-1)
        target_context = target_context[:, None, :].expand(batch_size, num_stations, -1)
        rental_input = torch.cat([rental_final, return_final, station_context, target_context], dim=-1)
        return_input = torch.cat([return_final, rental_final, station_context, target_context], dim=-1)
        rental_pred = self.rental_head(rental_input).squeeze(-1)
        return_pred = self.return_head(return_input).squeeze(-1)
        return torch.stack([rental_pred, return_pred], dim=-1)
