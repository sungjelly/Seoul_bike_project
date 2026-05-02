from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - depends on the training environment
    torch = None
    Dataset = object


class SeoulBikeLSTMDataset(Dataset):
    """Lazy dataset for the LSTM baseline arrays.

    Windows are sliced from saved base arrays on demand, so the full
    (num_samples, sequence_length, num_features) tensor is never materialized.
    """

    def __init__(
        self,
        data_dir: str | Path = "data/lstm_baseline",
        split: str = "train",
        sequence_length: int | None = None,
        horizon: int | None = None,
        mmap_mode: str | None = "r",
    ) -> None:
        if torch is None:
            raise ImportError(
                "PyTorch is required to use SeoulBikeLSTMDataset. Install it with: pip install torch"
            )

        self.torch = torch
        self.data_dir = Path(data_dir)
        self.split = split

        with (self.data_dir / "feature_config.json").open("r", encoding="utf-8") as f:
            self.feature_config = json.load(f)
        with (self.data_dir / "splits.json").open("r", encoding="utf-8") as f:
            splits = json.load(f)

        split_key = f"{split}_target_indices"
        if split_key not in splits:
            raise ValueError(f"Unknown split '{split}'. Expected one of: train, val, test")

        task = self.feature_config["task"]
        self.sequence_length = int(sequence_length or task["sequence_length"])
        self.horizon = int(horizon or task["horizon"])
        self.target_indices = np.asarray(splits[split_key], dtype=np.int64)

        self.dynamic_features = np.load(self.data_dir / "dynamic_features.npy", mmap_mode=mmap_mode)
        self.targets = np.load(self.data_dir / "targets.npy", mmap_mode=mmap_mode)
        self.targets_raw = np.load(self.data_dir / "targets_raw.npy", mmap_mode=mmap_mode)
        self.static_numeric = np.load(self.data_dir / "static_numeric.npy", mmap_mode=mmap_mode)
        self.district_ids = np.load(self.data_dir / "district_ids.npy", mmap_mode=mmap_mode)
        self.timestamps = np.load(self.data_dir / "timestamps.npy", mmap_mode=mmap_mode)

        self.num_stations = int(self.dynamic_features.shape[1])

    def __len__(self) -> int:
        return int(len(self.target_indices) * self.num_stations)

    def __getitem__(self, sample_idx: int) -> dict:
        if sample_idx < 0:
            sample_idx += len(self)
        if sample_idx < 0 or sample_idx >= len(self):
            raise IndexError(sample_idx)

        local_time_pos = sample_idx // self.num_stations
        station_idx = sample_idx % self.num_stations
        target_idx = int(self.target_indices[local_time_pos])

        input_end_idx = target_idx - self.horizon
        input_start_idx = input_end_idx - self.sequence_length + 1
        if input_start_idx < 0:
            raise IndexError(
                f"Invalid sample {sample_idx}: input_start_idx={input_start_idx}. "
                "Check sequence_length, horizon, and splits."
            )

        x_seq = self.dynamic_features[
            input_start_idx : input_end_idx + 1,
            station_idx,
            :,
        ]

        return {
            "x_seq": self.torch.as_tensor(np.array(x_seq, copy=True), dtype=self.torch.float32),
            "static_numeric": self.torch.as_tensor(
                np.array(self.static_numeric[station_idx], copy=True), dtype=self.torch.float32
            ),
            "station_index": self.torch.tensor(station_idx, dtype=self.torch.long),
            "district_id": self.torch.tensor(
                int(self.district_ids[station_idx]), dtype=self.torch.long
            ),
            "target": self.torch.as_tensor(
                np.array(self.targets[target_idx, station_idx, :], copy=True),
                dtype=self.torch.float32,
            ),
            "target_raw": self.torch.as_tensor(
                np.array(self.targets_raw[target_idx, station_idx, :], copy=True),
                dtype=self.torch.float32,
            ),
            "target_timestamp": self.timestamps[target_idx],
        }


LSTMBaselineDataset = SeoulBikeLSTMDataset
