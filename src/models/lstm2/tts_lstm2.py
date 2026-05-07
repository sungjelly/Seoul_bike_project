from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class TTSLSTM2(nn.Module):
    """Tri-Temporal Seasonal LSTM v2 for multi-horizon station demand prediction.

    LSTM branches consume only demand and weather sequence features. Calendar
    target-time features are concatenated into the final MLP head.
    """

    def __init__(
        self,
        input_dim: int,
        target_time_dim: int,
        static_numeric_dim: int,
        output_dim: int,
        horizon: int,
        num_stations: int,
        num_districts: int,
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
        mlp_hidden_dims: Sequence[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim != 7:
            raise ValueError(f"tts_lstm2 requires input_dim=7, got {input_dim}.")
        if target_time_dim != 8:
            raise ValueError(f"tts_lstm2 requires target_time_dim=8, got {target_time_dim}.")
        if horizon != 8:
            raise ValueError(f"tts_lstm2 requires horizon=8, got {horizon}.")
        if output_dim != 2:
            raise ValueError(f"tts_lstm2 requires output_dim=2, got {output_dim}.")

        self.input_dim = int(input_dim)
        self.target_time_dim = int(target_time_dim)
        self.static_numeric_dim = int(static_numeric_dim)
        self.output_dim = int(output_dim)
        self.horizon = int(horizon)

        self.register_buffer(
            "recent_indices",
            self._branch_indices(window_offsets, recent_offsets, "recent"),
            persistent=False,
        )
        self.register_buffer(
            "daily_indices",
            self._branch_indices(window_offsets, daily_offsets, "daily"),
            persistent=False,
        )
        self.register_buffer(
            "weekly_indices",
            self._branch_indices(window_offsets, weekly_offsets, "weekly"),
            persistent=False,
        )

        self.recent_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=recent_hidden_dim,
            num_layers=recent_num_layers,
            batch_first=True,
            dropout=dropout if recent_num_layers > 1 else 0.0,
        )
        self.daily_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=daily_hidden_dim,
            num_layers=daily_num_layers,
            batch_first=True,
            dropout=dropout if daily_num_layers > 1 else 0.0,
        )
        self.weekly_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=weekly_hidden_dim,
            num_layers=weekly_num_layers,
            batch_first=True,
            dropout=dropout if weekly_num_layers > 1 else 0.0,
        )
        self.station_embedding = nn.Embedding(num_stations, station_embedding_dim)
        self.district_embedding = nn.Embedding(num_districts, district_embedding_dim)

        concat_dim = (
            recent_hidden_dim
            + daily_hidden_dim
            + weekly_hidden_dim
            + station_embedding_dim
            + district_embedding_dim
            + static_numeric_dim
            + target_time_dim
        )
        self.head = self._make_mlp(concat_dim, horizon * output_dim, mlp_hidden_dims or [], dropout)

    @staticmethod
    def _branch_indices(
        window_offsets: Sequence[int],
        branch_offsets: Sequence[int],
        branch_name: str,
    ) -> torch.Tensor:
        if not branch_offsets:
            raise ValueError(f"{branch_name}_offsets must be non-empty.")
        offsets = [int(offset) for offset in window_offsets]
        branch = [int(offset) for offset in branch_offsets]
        non_negative = [offset for offset in branch if offset >= 0]
        if non_negative:
            raise ValueError(f"{branch_name}_offsets must be negative, got {non_negative}.")

        offset_to_index = {offset: idx for idx, offset in enumerate(offsets)}
        missing = [offset for offset in branch if offset not in offset_to_index]
        if missing:
            raise ValueError(
                f"{branch_name}_offsets contains offsets missing from window_offsets: {missing}"
            )
        return torch.tensor([offset_to_index[offset] for offset in branch], dtype=torch.long)

    @staticmethod
    def _make_mlp(
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            hidden = int(hidden_dim)
            if hidden <= 0:
                raise ValueError(f"mlp_hidden_dims must contain positive integers, got {hidden_dim}.")
            layers.extend([nn.Linear(prev_dim, hidden), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = hidden
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    def forward(
        self,
        x_seq: torch.Tensor,
        target_time_features: torch.Tensor,
        static_numeric: torch.Tensor,
        station_index: torch.Tensor,
        district_id: torch.Tensor,
        operation_type_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x_seq.ndim != 3:
            raise ValueError(f"x_seq must have rank 3, got shape {tuple(x_seq.shape)}.")
        if x_seq.shape[-1] != self.input_dim:
            raise ValueError(f"x_seq last dimension must be {self.input_dim}, got {x_seq.shape[-1]}.")
        if target_time_features is None:
            raise ValueError("target_time_features is required for tts_lstm2.")
        if target_time_features.ndim != 2:
            raise ValueError(
                f"target_time_features must have rank 2, got shape {tuple(target_time_features.shape)}."
            )
        if target_time_features.shape[-1] != self.target_time_dim:
            raise ValueError(
                f"target_time_features last dimension must be {self.target_time_dim}, "
                f"got {target_time_features.shape[-1]}."
            )
        if operation_type_id is not None:
            operation_type = operation_type_id.to(dtype=static_numeric.dtype).unsqueeze(1)
            static_numeric = torch.cat([static_numeric, operation_type], dim=1)
        if static_numeric.ndim != 2 or static_numeric.shape[-1] != self.static_numeric_dim:
            raise ValueError(
                f"static_numeric last dimension must be {self.static_numeric_dim}, "
                f"got shape {tuple(static_numeric.shape)}."
            )

        recent_seq = x_seq.index_select(1, self.recent_indices)
        daily_seq = x_seq.index_select(1, self.daily_indices)
        weekly_seq = x_seq.index_select(1, self.weekly_indices)

        _, (recent_hidden, _) = self.recent_lstm(recent_seq)
        _, (daily_hidden, _) = self.daily_lstm(daily_seq)
        _, (weekly_hidden, _) = self.weekly_lstm(weekly_seq)

        station_emb = self.station_embedding(station_index)
        district_emb = self.district_embedding(district_id)
        features = torch.cat(
            [
                recent_hidden[-1],
                daily_hidden[-1],
                weekly_hidden[-1],
                station_emb,
                district_emb,
                static_numeric,
                target_time_features,
            ],
            dim=1,
        )
        output = self.head(features)
        batch_size = int(x_seq.shape[0])
        expected = self.horizon * self.output_dim
        if output.shape != (batch_size, expected):
            raise ValueError(f"tts_lstm2 head output must have shape ({batch_size}, {expected}), got {tuple(output.shape)}.")
        try:
            return output.reshape(batch_size, self.horizon, self.output_dim)
        except RuntimeError as exc:
            raise ValueError(
                f"Could not reshape output to ({batch_size}, {self.horizon}, {self.output_dim})."
            ) from exc
