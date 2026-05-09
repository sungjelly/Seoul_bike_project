from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object


GRAPH_FILES = [
    "neighbor_index_rr.npy",
    "neighbor_index_dd.npy",
    "neighbor_index_rd.npy",
    "neighbor_index_dr.npy",
    "edge_attr_rr.npy",
    "edge_attr_dd.npy",
    "edge_attr_rd.npy",
    "edge_attr_dr.npy",
]


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_graph_dir(data_dir: Path) -> Path:
    summary_path = data_dir / "dataset_summary.json"
    if summary_path.exists():
        graph_dir = Path(str(read_json(summary_path)["graph_dir"]).replace("\\", "/"))
        if not graph_dir.is_absolute():
            graph_dir = Path.cwd() / graph_dir
        return graph_dir
    return data_dir / "graph"


def validate_graph_files(graph_dir: Path, num_stations: int | None = None) -> dict[str, np.ndarray]:
    missing = [name for name in GRAPH_FILES if not (graph_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing graph files in {graph_dir}: {missing}")
    graph = {name[:-4]: np.load(graph_dir / name, mmap_mode="r") for name in GRAPH_FILES}
    for key, array in graph.items():
        if "edge_attr" in key and not np.isfinite(array).all():
            raise ValueError(f"{key} contains NaN or inf.")
        if "neighbor_index" in key:
            if array.ndim != 2:
                raise ValueError(f"{key} must have shape S x K.")
            if num_stations is not None and array.shape[0] != num_stations:
                raise ValueError(f"{key} station count {array.shape[0]} does not match dataset S={num_stations}.")
            if int(array.min()) < 0 or int(array.max()) >= array.shape[0]:
                raise ValueError(f"{key} contains out-of-range station indices.")
    return graph


class TCTGATGraphSnapshotDataset(Dataset):
    """One item is one target timestamp containing all stations."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        mmap_mode: str | None = "r",
        return_raw_target: bool = True,
        validate_graph: bool = True,
    ) -> None:
        if torch is None:
            raise ImportError("PyTorch is required to use TCTGATGraphSnapshotDataset.")
        self.torch = torch
        self.data_dir = Path(data_dir)
        self.split = split
        self.return_raw_target = bool(return_raw_target)

        self.rental_features = np.load(self.data_dir / "rental_features.npy", mmap_mode=mmap_mode)
        self.return_features = np.load(self.data_dir / "return_features.npy", mmap_mode=mmap_mode)
        self.targets = np.load(self.data_dir / "targets.npy", mmap_mode=mmap_mode)
        self.targets_raw = np.load(self.data_dir / "targets_raw.npy", mmap_mode=mmap_mode) if return_raw_target else None
        self.future_weather_features = np.load(self.data_dir / "future_weather_features.npy", mmap_mode=mmap_mode)
        self.target_time_features = np.load(self.data_dir / "target_time_features.npy", mmap_mode=mmap_mode)
        self.static_numeric = np.load(self.data_dir / "static_numeric.npy", mmap_mode=mmap_mode)
        self.district_ids = np.load(self.data_dir / "district_ids.npy", mmap_mode=mmap_mode)
        self.operation_type_ids = np.load(self.data_dir / "operation_type_ids.npy", mmap_mode=mmap_mode)
        self.station_numbers = np.load(self.data_dir / "station_numbers.npy", mmap_mode=mmap_mode)
        self.timestamps = np.load(self.data_dir / "timestamps.npy", mmap_mode=mmap_mode)
        self.window_offsets = np.load(self.data_dir / "window_offsets.npy").astype(np.int64)
        self.sample_time_index = np.load(self.data_dir / f"sample_time_index_{split}.npy", mmap_mode=mmap_mode)
        self.feature_config = read_json(self.data_dir / "feature_config.json")
        self.graph_dir = resolve_graph_dir(self.data_dir)
        self.graph = validate_graph_files(self.graph_dir, self.rental_features.shape[1]) if validate_graph else {}
        self._validate()

    def _validate(self) -> None:
        if self.sample_time_index.ndim != 1:
            raise ValueError(f"sample_time_index_{self.split}.npy must be 1D.")
        if self.rental_features.ndim != 3 or self.rental_features.shape[-1] != 5:
            raise ValueError("rental_features.npy must have shape T x S x 5.")
        if self.return_features.ndim != 3 or self.return_features.shape[-1] != 5:
            raise ValueError("return_features.npy must have shape T x S x 5.")
        if self.targets.ndim != 3 or self.targets.shape[-1] != 2:
            raise ValueError("targets.npy must have shape T x S x 2.")
        if self.future_weather_features.ndim != 2 or self.future_weather_features.shape[-1] != 4:
            raise ValueError("future_weather_features.npy must have shape T x 4.")
        if self.target_time_features.ndim != 2 or self.target_time_features.shape[-1] != 8:
            raise ValueError("target_time_features.npy must have shape T x 8.")
        if "net_demand" in json.dumps(self.feature_config, ensure_ascii=False):
            raise ValueError("TCT-GAT feature_config must not contain net_demand.")
        T, S, _ = self.rental_features.shape
        if self.return_features.shape[:2] != (T, S) or self.targets.shape[:2] != (T, S):
            raise ValueError("Core arrays must share T and S dimensions.")
        if len(self.sample_time_index):
            input_idx = self.sample_time_index[:, None].astype(np.int64) + self.window_offsets[None, :]
            if int(input_idx.min()) < 0 or int(input_idx.max()) >= T:
                raise ValueError("sample_time_index and window_offsets produce out-of-bounds inputs.")

    def __len__(self) -> int:
        return int(len(self.sample_time_index))

    def __getitem__(self, idx: int) -> dict:
        if idx < 0:
            idx += len(self)
        target_time_idx = int(self.sample_time_index[idx])
        input_time_idx = target_time_idx + self.window_offsets
        item = {
            "rental_seq": self.torch.as_tensor(np.array(self.rental_features[input_time_idx, :, :], copy=True).transpose(1, 0, 2), dtype=self.torch.float32),
            "return_seq": self.torch.as_tensor(np.array(self.return_features[input_time_idx, :, :], copy=True).transpose(1, 0, 2), dtype=self.torch.float32),
            "y": self.torch.as_tensor(np.array(self.targets[target_time_idx, :, :], copy=True), dtype=self.torch.float32),
            "target_time_features": self.torch.as_tensor(np.array(self.target_time_features[target_time_idx, :], copy=True), dtype=self.torch.float32),
            "future_weather_features": self.torch.as_tensor(np.array(self.future_weather_features[target_time_idx, :], copy=True), dtype=self.torch.float32),
            "target_time_idx": self.torch.tensor(target_time_idx, dtype=self.torch.long),
            "station_index": self.torch.arange(self.rental_features.shape[1], dtype=self.torch.long),
            "static_numeric": self.torch.as_tensor(np.array(self.static_numeric, copy=True), dtype=self.torch.float32),
            "district_id": self.torch.as_tensor(np.array(self.district_ids, copy=True), dtype=self.torch.long),
            "operation_type_id": self.torch.as_tensor(np.array(self.operation_type_ids, copy=True), dtype=self.torch.long),
        }
        if self.return_raw_target and self.targets_raw is not None:
            item["y_raw"] = self.torch.as_tensor(np.array(self.targets_raw[target_time_idx, :, :], copy=True), dtype=self.torch.float32)
        return item


class FastTCTGATBatchBuilder:
    """Vectorized graph-snapshot batch iterator."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        batch_size: int,
        device: str | "torch.device" = "cpu",
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int | None = None,
        return_raw_target: bool = False,
        mmap_mode: str | None = "r",
        validate_graph: bool = True,
    ) -> None:
        if torch is None:
            raise ImportError("PyTorch is required to use FastTCTGATBatchBuilder.")
        self.torch = torch
        self.data_dir = Path(data_dir)
        self.split = split
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.return_raw_target = bool(return_raw_target)
        self.rng = np.random.default_rng(seed)

        self.rental_features = np.load(self.data_dir / "rental_features.npy", mmap_mode=mmap_mode)
        self.return_features = np.load(self.data_dir / "return_features.npy", mmap_mode=mmap_mode)
        self.targets = np.load(self.data_dir / "targets.npy", mmap_mode=mmap_mode)
        self.targets_raw = np.load(self.data_dir / "targets_raw.npy", mmap_mode=mmap_mode) if return_raw_target else None
        self.future_weather_features = np.load(self.data_dir / "future_weather_features.npy", mmap_mode=mmap_mode)
        self.target_time_features = np.load(self.data_dir / "target_time_features.npy", mmap_mode=mmap_mode)
        self.static_numeric = np.load(self.data_dir / "static_numeric.npy", mmap_mode=mmap_mode)
        self.district_ids = np.load(self.data_dir / "district_ids.npy", mmap_mode=mmap_mode)
        self.operation_type_ids = np.load(self.data_dir / "operation_type_ids.npy", mmap_mode=mmap_mode)
        self.station_numbers = np.load(self.data_dir / "station_numbers.npy", mmap_mode=mmap_mode)
        self.timestamps = np.load(self.data_dir / "timestamps.npy", mmap_mode=mmap_mode)
        self.window_offsets = np.load(self.data_dir / "window_offsets.npy").astype(np.int64)
        self.sample_time_index = np.load(self.data_dir / f"sample_time_index_{split}.npy", mmap_mode=mmap_mode)
        self.feature_config = read_json(self.data_dir / "feature_config.json")
        self.graph_dir = resolve_graph_dir(self.data_dir)
        self.graph = validate_graph_files(self.graph_dir, self.rental_features.shape[1]) if validate_graph else {}
        self._validate()

    def _validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.sample_time_index.ndim != 1:
            raise ValueError(f"sample_time_index_{self.split}.npy must be 1D.")
        if self.rental_features.shape[-1] != 5 or self.return_features.shape[-1] != 5:
            raise ValueError("TCT-GAT sequence feature dim must be 5.")
        if self.targets.shape[-1] != 2:
            raise ValueError("TCT-GAT target dim must be 2.")
        if self.target_time_features.shape[-1] != 8:
            raise ValueError("target_time_features dim must be 8.")
        if self.future_weather_features.shape[-1] != 4:
            raise ValueError("future_weather_features dim must be 4.")
        if "net_demand" in json.dumps(self.feature_config, ensure_ascii=False):
            raise ValueError("TCT-GAT feature_config must not contain net_demand.")

    def __len__(self) -> int:
        full, rem = divmod(len(self.sample_time_index), self.batch_size)
        return full if self.drop_last or rem == 0 else full + 1

    def __iter__(self) -> Iterator[dict]:
        order = np.arange(len(self.sample_time_index), dtype=np.int64)
        if self.shuffle:
            self.rng.shuffle(order)
        end_limit = len(order)
        if self.drop_last:
            end_limit = (len(order) // self.batch_size) * self.batch_size
        for start in range(0, end_limit, self.batch_size):
            ids = order[start : start + self.batch_size]
            if len(ids):
                yield self._build_batch(ids)

    def _to_tensor(self, array: np.ndarray, dtype: "torch.dtype") -> "torch.Tensor":
        return self.torch.as_tensor(np.array(array, copy=True), dtype=dtype, device=self.device)

    def _build_batch(self, batch_ids: np.ndarray) -> dict:
        target_idx = self.sample_time_index[batch_ids].astype(np.int64)
        input_idx = target_idx[:, None] + self.window_offsets[None, :]
        S = self.rental_features.shape[1]
        batch = {
            "rental_seq": self._to_tensor(self.rental_features[input_idx, :, :].transpose(0, 2, 1, 3), self.torch.float32),
            "return_seq": self._to_tensor(self.return_features[input_idx, :, :].transpose(0, 2, 1, 3), self.torch.float32),
            "y": self._to_tensor(self.targets[target_idx, :, :], self.torch.float32),
            "target_time_features": self._to_tensor(self.target_time_features[target_idx, :], self.torch.float32),
            "future_weather_features": self._to_tensor(self.future_weather_features[target_idx, :], self.torch.float32),
            "target_time_idx": self._to_tensor(target_idx, self.torch.long),
            "station_index": self._to_tensor(np.arange(S, dtype=np.int64), self.torch.long),
            "static_numeric": self._to_tensor(self.static_numeric, self.torch.float32),
            "district_id": self._to_tensor(self.district_ids, self.torch.long),
            "operation_type_id": self._to_tensor(self.operation_type_ids, self.torch.long),
        }
        if self.return_raw_target and self.targets_raw is not None:
            batch["y_raw"] = self._to_tensor(self.targets_raw[target_idx, :, :], self.torch.float32)
        for key in ["rental_seq", "return_seq", "y", "target_time_features", "future_weather_features"]:
            if not self.torch.isfinite(batch[key]).all():
                raise ValueError(f"Batch contains non-finite values in {key}.")
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-check a TCT-GAT graph-snapshot dataset batch.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/tct_gat_processed/tct_gat1_ar"))
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batches = FastTCTGATBatchBuilder(args.data_dir, args.split, args.batch_size, shuffle=False, return_raw_target=True)
    batch = next(iter(batches))
    print(f"rental_seq.shape={tuple(batch['rental_seq'].shape)}")
    print(f"return_seq.shape={tuple(batch['return_seq'].shape)}")
    print(f"y.shape={tuple(batch['y'].shape)}")
    print(f"target_time_features.shape={tuple(batch['target_time_features'].shape)}")
    print(f"future_weather_features.shape={tuple(batch['future_weather_features'].shape)}")
    print(f"static_numeric.shape={tuple(batch['static_numeric'].shape)}")
    print(f"district_id.shape={tuple(batch['district_id'].shape)}")
    print(f"operation_type_id.shape={tuple(batch['operation_type_id'].shape)}")
    for name, array in batches.graph.items():
        print(f"{name}.shape={tuple(array.shape)}")


if __name__ == "__main__":
    main()
