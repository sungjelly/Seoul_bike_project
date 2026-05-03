from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - depends on runtime environment
    torch = None
    Dataset = object


def resolve_array_dir(data_dir: Path) -> Path:
    """Return the directory that stores the large shared array files."""
    for metadata_name in ["base_data.json", "dataset_summary.json"]:
        metadata_path = data_dir / metadata_name
        if not metadata_path.exists():
            continue
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        base_data_dir = metadata.get("base_data_dir")
        if base_data_dir:
            path = Path(str(base_data_dir).replace("\\", "/"))
            if not path.is_absolute():
                path = data_dir / path
            return path.resolve()
    return data_dir


class SeoulBikeLSTMDataset(Dataset):
    """PyTorch Dataset for compact Seoul bike LSTM arrays.

    Samples are stored as [target_time_idx, station_idx] pairs. Input windows are
    gathered with window_offsets, so windows can be continuous, sparse, daily, or
    weekly without changing this reader.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        mmap_mode: str | None = "r",
        return_raw_target: bool = True,
        return_static: bool = True,
        return_metadata: bool = False,
    ) -> None:
        if torch is None:
            raise ImportError("PyTorch is required to use SeoulBikeLSTMDataset.")

        self.torch = torch
        self.data_dir = Path(data_dir)
        self.array_dir = resolve_array_dir(self.data_dir)
        self.split = split
        self.return_raw_target = return_raw_target
        self.return_static = return_static
        self.return_metadata = return_metadata

        self.dynamic_features = np.load(self.array_dir / "dynamic_features.npy", mmap_mode=mmap_mode)
        self.targets = np.load(self.array_dir / "targets.npy", mmap_mode=mmap_mode)
        self.targets_raw = self._load_optional("targets_raw.npy", mmap_mode) if return_raw_target else None
        self.static_numeric = self._load_optional("static_numeric.npy", mmap_mode) if return_static else None
        self.district_ids = self._load_optional("district_ids.npy", mmap_mode) if return_static else None
        self.operation_type_ids = self._load_optional("operation_type_ids.npy", mmap_mode) if return_static else None
        self.station_numbers = self._load_optional("station_numbers.npy", mmap_mode) if return_metadata else None
        self.timestamps = self._load_optional("timestamps.npy", mmap_mode) if return_metadata else None
        self.window_offsets = np.load(self.data_dir / "window_offsets.npy").astype(np.int64)
        self.sample_index = np.load(self.data_dir / f"sample_index_{split}.npy", mmap_mode=mmap_mode)

        self._validate()

    def _load_optional(self, filename: str, mmap_mode: str | None) -> np.ndarray | None:
        path = self.array_dir / filename
        if not path.exists():
            return None
        return np.load(path, mmap_mode=mmap_mode)

    def _validate(self) -> None:
        if self.sample_index.ndim != 2 or self.sample_index.shape[1] != 2:
            raise ValueError(f"sample_index_{self.split}.npy must have shape (N, 2).")
        if self.window_offsets.ndim != 1 or len(self.window_offsets) == 0:
            raise ValueError("window_offsets.npy must be a non-empty 1D array.")
        if self.dynamic_features.ndim != 3:
            raise ValueError("dynamic_features.npy must have shape (T, S, F).")
        if self.targets.ndim != 3:
            raise ValueError("targets.npy must have shape (T, S, target_dim).")
        T, S, _ = self.dynamic_features.shape
        if self.targets.shape[:2] != (T, S):
            raise ValueError("targets.npy first two dimensions must match dynamic_features.npy.")
        if len(self.sample_index) > 0:
            target_time_idx = self.sample_index[:, 0].astype(np.int64)
            station_idx = self.sample_index[:, 1].astype(np.int64)
            if target_time_idx.min() < 0 or target_time_idx.max() >= T:
                raise ValueError("sample index target_time_idx is out of bounds.")
            if station_idx.min() < 0 or station_idx.max() >= S:
                raise ValueError("sample index station_idx is out of bounds.")
            input_time_idx = target_time_idx[:, None] + self.window_offsets[None, :]
            if input_time_idx.min() < 0 or input_time_idx.max() >= T:
                raise ValueError("sample index and window_offsets produce out-of-bounds input times.")

    def __len__(self) -> int:
        return int(len(self.sample_index))

    def __getitem__(self, idx: int) -> dict:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        target_time_idx = int(self.sample_index[idx, 0])
        station_idx = int(self.sample_index[idx, 1])
        input_time_idx = target_time_idx + self.window_offsets

        x_np = self.dynamic_features[input_time_idx, station_idx, :]
        y_np = self.targets[target_time_idx, station_idx, :]

        item = {
            "x": self.torch.as_tensor(np.array(x_np, copy=True), dtype=self.torch.float32),
            "y": self.torch.as_tensor(np.array(y_np, copy=True), dtype=self.torch.float32),
            "target_time_idx": self.torch.tensor(target_time_idx, dtype=self.torch.long),
            "station_idx": self.torch.tensor(station_idx, dtype=self.torch.long),
        }

        if self.return_raw_target and self.targets_raw is not None:
            item["y_raw"] = self.torch.as_tensor(
                np.array(self.targets_raw[target_time_idx, station_idx, :], copy=True),
                dtype=self.torch.float32,
            )
        if self.return_static:
            if self.static_numeric is not None:
                item["static_numeric"] = self.torch.as_tensor(
                    np.array(self.static_numeric[station_idx], copy=True),
                    dtype=self.torch.float32,
                )
            if self.district_ids is not None:
                item["district_id"] = self.torch.tensor(int(self.district_ids[station_idx]), dtype=self.torch.long)
            if self.operation_type_ids is not None:
                item["operation_type_id"] = self.torch.tensor(
                    int(self.operation_type_ids[station_idx]),
                    dtype=self.torch.long,
                )
        if self.return_metadata:
            if self.timestamps is not None:
                item["target_timestamp"] = self.timestamps[target_time_idx]
            if self.station_numbers is not None:
                item["station_number"] = int(self.station_numbers[station_idx])
        return item


class FastLSTMBatchBuilder:
    """Vectorized batch iterator for large-batch LSTM training."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        batch_size: int,
        device: str | "torch.device" = "cpu",
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int | None = None,
        return_static: bool = True,
        return_raw_target: bool = False,
        mmap_mode: str | None = "r",
    ) -> None:
        if torch is None:
            raise ImportError("PyTorch is required to use FastLSTMBatchBuilder.")

        self.torch = torch
        self.data_dir = Path(data_dir)
        self.array_dir = resolve_array_dir(self.data_dir)
        self.split = split
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.return_static = bool(return_static)
        self.return_raw_target = bool(return_raw_target)
        self.rng = np.random.default_rng(seed)

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self.dynamic_features = np.load(self.array_dir / "dynamic_features.npy", mmap_mode=mmap_mode)
        self.targets = np.load(self.array_dir / "targets.npy", mmap_mode=mmap_mode)
        self.targets_raw = self._load_optional("targets_raw.npy", mmap_mode) if return_raw_target else None
        self.static_numeric = self._load_optional("static_numeric.npy", mmap_mode) if return_static else None
        self.district_ids = self._load_optional("district_ids.npy", mmap_mode) if return_static else None
        self.operation_type_ids = self._load_optional("operation_type_ids.npy", mmap_mode) if return_static else None
        self.window_offsets = np.load(self.data_dir / "window_offsets.npy").astype(np.int64)
        self.sample_index = np.load(self.data_dir / f"sample_index_{split}.npy", mmap_mode=mmap_mode)
        self._validate()

    def _load_optional(self, filename: str, mmap_mode: str | None) -> np.ndarray | None:
        path = self.array_dir / filename
        if not path.exists():
            return None
        return np.load(path, mmap_mode=mmap_mode)

    def _validate(self) -> None:
        if self.sample_index.ndim != 2 or self.sample_index.shape[1] != 2:
            raise ValueError(f"sample_index_{self.split}.npy must have shape (N, 2).")
        if self.dynamic_features.ndim != 3:
            raise ValueError("dynamic_features.npy must have shape (T, S, F).")
        if self.targets.shape[:2] != self.dynamic_features.shape[:2]:
            raise ValueError("targets.npy first two dimensions must match dynamic_features.npy.")

    def __len__(self) -> int:
        full_batches, remainder = divmod(len(self.sample_index), self.batch_size)
        if remainder and not self.drop_last:
            return full_batches + 1
        return full_batches

    def __iter__(self) -> Iterator[dict]:
        order = np.arange(len(self.sample_index), dtype=np.int64)
        if self.shuffle:
            self.rng.shuffle(order)

        end_limit = len(order)
        if self.drop_last:
            end_limit = (len(order) // self.batch_size) * self.batch_size

        for start in range(0, end_limit, self.batch_size):
            batch_ids = order[start : start + self.batch_size]
            if len(batch_ids) == 0:
                continue
            yield self._build_batch(batch_ids)

    def _to_tensor(self, array: np.ndarray, dtype: "torch.dtype") -> "torch.Tensor":
        tensor = self.torch.as_tensor(np.array(array, copy=True), dtype=dtype)
        return tensor.to(self.device, non_blocking=True)

    def _build_batch(self, batch_ids: np.ndarray) -> dict:
        pairs = self.sample_index[batch_ids]
        target_time_idx = pairs[:, 0].astype(np.int64)
        station_idx = pairs[:, 1].astype(np.int64)
        input_time_idx = target_time_idx[:, None] + self.window_offsets[None, :]

        x_np = self.dynamic_features[input_time_idx, station_idx[:, None], :]
        y_np = self.targets[target_time_idx, station_idx, :]

        batch = {
            "x": self._to_tensor(x_np, self.torch.float32),
            "y": self._to_tensor(y_np, self.torch.float32),
            "target_time_idx": self._to_tensor(target_time_idx, self.torch.long),
            "station_idx": self._to_tensor(station_idx, self.torch.long),
        }

        if self.return_static:
            if self.static_numeric is not None:
                batch["static_numeric"] = self._to_tensor(self.static_numeric[station_idx], self.torch.float32)
            if self.district_ids is not None:
                batch["district_id"] = self._to_tensor(self.district_ids[station_idx], self.torch.long)
            if self.operation_type_ids is not None:
                batch["operation_type_id"] = self._to_tensor(
                    self.operation_type_ids[station_idx],
                    self.torch.long,
                )
        if self.return_raw_target and self.targets_raw is not None:
            batch["y_raw"] = self._to_tensor(self.targets_raw[target_time_idx, station_idx, :], self.torch.float32)
        return batch


LSTMBaselineDataset = SeoulBikeLSTMDataset
