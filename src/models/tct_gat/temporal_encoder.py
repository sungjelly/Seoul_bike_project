from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class TriScaleTemporalEncoder(nn.Module):
    """Encode sparse recent/daily/weekly station histories with separate LSTMs."""

    def __init__(
        self,
        input_dim: int,
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
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.temporal_dim = int(recent_hidden_dim + daily_hidden_dim + weekly_hidden_dim)
        self.register_buffer("recent_indices", self._branch_indices(window_offsets, recent_offsets, "recent"), persistent=False)
        self.register_buffer("daily_indices", self._branch_indices(window_offsets, daily_offsets, "daily"), persistent=False)
        self.register_buffer("weekly_indices", self._branch_indices(window_offsets, weekly_offsets, "weekly"), persistent=False)
        self.recent_lstm = nn.LSTM(
            input_dim,
            recent_hidden_dim,
            num_layers=recent_num_layers,
            batch_first=True,
            dropout=dropout if recent_num_layers > 1 else 0.0,
        )
        self.daily_lstm = nn.LSTM(
            input_dim,
            daily_hidden_dim,
            num_layers=daily_num_layers,
            batch_first=True,
            dropout=dropout if daily_num_layers > 1 else 0.0,
        )
        self.weekly_lstm = nn.LSTM(
            input_dim,
            weekly_hidden_dim,
            num_layers=weekly_num_layers,
            batch_first=True,
            dropout=dropout if weekly_num_layers > 1 else 0.0,
        )

    @staticmethod
    def _branch_indices(window_offsets: Sequence[int], branch_offsets: Sequence[int], name: str) -> torch.Tensor:
        if not branch_offsets:
            raise ValueError(f"{name}_offsets must be non-empty.")
        offset_to_index = {int(offset): idx for idx, offset in enumerate(window_offsets)}
        missing = [int(offset) for offset in branch_offsets if int(offset) not in offset_to_index]
        if missing:
            raise ValueError(f"{name}_offsets missing from window_offsets: {missing}")
        return torch.tensor([offset_to_index[int(offset)] for offset in branch_offsets], dtype=torch.long)

    def _encode_branch(self, x: torch.Tensor, indices: torch.Tensor, lstm: nn.LSTM) -> torch.Tensor:
        branch = x.index_select(2, indices)
        batch, stations, steps, features = branch.shape
        branch = branch.reshape(batch * stations, steps, features)
        _, (hidden, _) = lstm(branch)
        return hidden[-1].reshape(batch, stations, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"x must have shape B x S x W x F, got {tuple(x.shape)}.")
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"x last dim must be {self.input_dim}, got {x.shape[-1]}.")
        return torch.cat(
            [
                self._encode_branch(x, self.recent_indices, self.recent_lstm),
                self._encode_branch(x, self.daily_indices, self.daily_lstm),
                self._encode_branch(x, self.weekly_indices, self.weekly_lstm),
            ],
            dim=-1,
        )
