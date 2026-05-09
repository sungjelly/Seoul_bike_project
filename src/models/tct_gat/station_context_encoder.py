from __future__ import annotations

import torch
from torch import nn


class StationContextEncoder(nn.Module):
    """Station, district, operation type, and numeric static context encoder."""

    def __init__(
        self,
        num_stations: int,
        num_districts: int,
        num_operation_types: int,
        static_numeric_dim: int,
        station_embedding_dim: int = 16,
        district_embedding_dim: int = 8,
        operation_type_embedding_dim: int = 1,
        static_context_dim: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.static_numeric_dim = int(static_numeric_dim)
        self.static_context_dim = int(static_context_dim)
        self.station_embedding = nn.Embedding(num_stations, station_embedding_dim)
        self.district_embedding = nn.Embedding(num_districts, district_embedding_dim)
        self.operation_type_embedding = nn.Embedding(max(num_operation_types, 1), operation_type_embedding_dim)
        input_dim = station_embedding_dim + district_embedding_dim + operation_type_embedding_dim + static_numeric_dim
        self.projection = nn.Sequential(
            nn.Linear(input_dim, static_context_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(static_context_dim, static_context_dim),
        )

    @staticmethod
    def _expand_static(value: torch.Tensor, batch_size: int, dtype: torch.dtype | None = None) -> torch.Tensor:
        if value.ndim == 1:
            value = value.unsqueeze(0).expand(batch_size, -1)
        elif value.ndim == 2 and value.shape[0] != batch_size:
            value = value.unsqueeze(0).expand(batch_size, -1, -1)
        if dtype is not None:
            value = value.to(dtype=dtype)
        return value

    def forward(
        self,
        static_numeric: torch.Tensor,
        station_index: torch.Tensor,
        district_id: torch.Tensor,
        operation_type_id: torch.Tensor | None = None,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        if static_numeric.ndim == 2:
            if batch_size is None:
                raise ValueError("batch_size is required when static_numeric has shape S x F.")
            static_numeric = static_numeric.unsqueeze(0).expand(batch_size, -1, -1)
        elif static_numeric.ndim != 3:
            raise ValueError(f"static_numeric must have shape S x F or B x S x F, got {tuple(static_numeric.shape)}.")
        batch_size = int(static_numeric.shape[0])
        station_index = self._expand_static(station_index, batch_size).to(device=static_numeric.device, dtype=torch.long)
        district_id = self._expand_static(district_id, batch_size).to(device=static_numeric.device, dtype=torch.long)
        if operation_type_id is None:
            operation_type_id = torch.zeros_like(district_id)
        else:
            operation_type_id = self._expand_static(operation_type_id, batch_size).to(device=static_numeric.device, dtype=torch.long)
        features = torch.cat(
            [
                self.station_embedding(station_index),
                self.district_embedding(district_id),
                self.operation_type_embedding(operation_type_id.clamp_min(0)),
                static_numeric.to(dtype=torch.float32),
            ],
            dim=-1,
        )
        return self.projection(features)
