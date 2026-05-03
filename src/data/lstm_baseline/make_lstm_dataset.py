from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    import yaml
except ImportError as exc:  # pragma: no cover - checked at runtime
    raise ImportError("PyYAML is required. Install it with: pip install pyyaml") from exc

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.data.lstm_baseline.scaling import apply_transform, standardize


DYNAMIC_FEATURE_COLUMNS = [
    "rental_count",
    "return_count",
    "net_demand",
    "avg_duration_min",
    "avg_distance_m",
    "temperature",
    "wind_speed",
    "rainfall",
    "humidity",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "is_holiday",
]

COUNT_COLUMNS = ["rental_count", "return_count"]
TARGET_COLUMNS_DEFAULT = ["rental_count", "return_count"]
CONTINUOUS_DYNAMIC_COLUMNS = [
    "avg_duration_min",
    "avg_distance_m",
    "temperature",
    "wind_speed",
    "rainfall",
    "humidity",
]
STATIC_NUMERIC_COLUMNS = ["latitude", "longitude", "dock_count_raw"]
STATIC_CATEGORICAL_COLUMNS = ["district_id", "operation_type_id"]
PANEL_REQUIRED_COLUMNS = [
    "timestamp",
    "station_number",
    "rental_count",
    "return_count",
    "net_demand",
    "avg_duration_min",
    "avg_distance_m",
]
WEATHER_REQUIRED_COLUMNS = ["timestamp", "temperature", "wind_speed", "rainfall", "humidity"]
STATION_REQUIRED_COLUMNS = [
    "station_number",
    "district",
    "latitude",
    "longitude",
    "dock_count_raw",
    "operation_type",
]


@dataclass(frozen=True)
class SourceSpec:
    name: str
    path: Path


@dataclass(frozen=True)
class SourceBoundary:
    name: str
    start_idx: int
    end_idx: int
    start_timestamp: str
    end_timestamp: str

    @property
    def length(self) -> int:
        return self.end_idx - self.start_idx + 1


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compact LSTM tensors from preprocessed panels.")
    parser.add_argument("--config", type=Path, default=Path("configs/lstm_dataset.yaml"))
    parser.add_argument("--preprocessed-root", type=Path)
    parser.add_argument("--station-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dataset-name", type=str)
    return parser.parse_args()


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = dict(config)
    if args.preprocessed_root is not None:
        config["preprocessed_root"] = str(args.preprocessed_root)
    if args.station_dir is not None:
        config["station_dir"] = str(args.station_dir)
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    if args.dataset_name is not None:
        config["dataset_name"] = args.dataset_name
    return config


def resolve_output_dir(config: dict[str, Any]) -> Path:
    output_dir = Path(config.get("output_dir", "data/lstm_processed"))
    dataset_name = config.get("dataset_name")
    if dataset_name and output_dir.name != dataset_name:
        output_dir = output_dir / str(dataset_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def resolve_sources(config: dict[str, Any]) -> list[SourceSpec]:
    root = Path(config.get("preprocessed_root", "data/preprocessed"))
    sources = config.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Config must define a non-empty sources list.")
    resolved: list[SourceSpec] = []
    seen: set[str] = set()
    for item in sources:
        if not isinstance(item, dict) or "name" not in item:
            raise ValueError("Each source must define at least a name.")
        name = str(item["name"])
        if name in seen:
            raise ValueError(f"Duplicate source name: {name}")
        seen.add(name)
        path = Path(item.get("path", root / name))
        resolved.append(SourceSpec(name=name, path=path))
    return resolved


def load_station_inputs(station_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    metadata_path = station_dir / "station_metadata_clean.parquet"
    station_numbers_path = station_dir / "station_numbers.npy"
    missing = [str(path) for path in [metadata_path, station_numbers_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing shared station inputs: {missing}")

    metadata = pd.read_parquet(metadata_path)
    missing_columns = set(STATION_REQUIRED_COLUMNS) - set(metadata.columns)
    if missing_columns:
        raise ValueError(f"Station metadata missing columns: {sorted(missing_columns)}")
    station_numbers = np.load(station_numbers_path).astype(np.int64)
    metadata_numbers = metadata["station_number"].astype(np.int64).to_numpy()
    if not np.array_equal(metadata_numbers, station_numbers):
        raise ValueError("station_metadata_clean.parquet order must match station_numbers.npy.")
    return metadata.reset_index(drop=True), station_numbers


def build_window_offsets(window_config: dict[str, Any]) -> np.ndarray:
    mode = str(window_config.get("mode", "explicit_offsets"))
    offsets: list[int] = []
    if mode == "explicit_offsets":
        offsets = [int(value) for value in window_config.get("offsets", [])]
    elif mode == "blocks":
        blocks = window_config.get("blocks", [])
        if not isinstance(blocks, list) or not blocks:
            raise ValueError("window.mode=blocks requires a non-empty blocks list.")
        for block in blocks:
            block_type = str(block.get("type"))
            length = int(block["length"])
            end_offset = int(block["end_offset"])
            if length <= 0:
                raise ValueError(f"Window block length must be positive: {block}")
            if block_type == "continuous":
                start = end_offset - length + 1
                offsets.extend(range(start, end_offset + 1))
            elif block_type == "strided":
                stride = int(block["stride"])
                if stride <= 0:
                    raise ValueError(f"Strided window block stride must be positive: {block}")
                start = end_offset - stride * (length - 1)
                offsets.extend(range(start, end_offset + 1, stride))
            else:
                raise ValueError(f"Unknown window block type: {block_type}")
    else:
        raise ValueError(f"Unknown window mode: {mode}")

    if not offsets:
        raise ValueError("Window offsets cannot be empty.")
    return np.asarray(sorted(set(offsets)), dtype=np.int32)


def validate_window_offsets(offsets: np.ndarray, horizon: int) -> None:
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if np.max(offsets) > -horizon:
        raise ValueError(
            f"max(window_offsets)={int(np.max(offsets))} violates horizon={horizon}. "
            f"The latest allowed input offset is {-horizon}."
        )
    if np.any(offsets >= 0):
        raise ValueError("All window offsets must be negative.")


def source_timestamps(source: SourceSpec) -> np.ndarray:
    summary_path = source.path / "preprocessing_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing preprocessing summary for source {source.name}: {summary_path}")
    summary = read_json(summary_path)
    timestamps = pd.date_range(
        summary["start_timestamp"],
        summary["end_timestamp"],
        freq="30min",
    ).to_numpy(dtype="datetime64[ns]")
    expected = int(summary["num_timestamps"])
    if len(timestamps) != expected:
        raise ValueError(f"{source.name}: expected {expected} timestamps, got {len(timestamps)}")
    validate_timestamps(source.name, timestamps)
    return timestamps


def validate_timestamps(source_name: str, timestamps: np.ndarray) -> None:
    ts = pd.DatetimeIndex(timestamps)
    if not ts.is_monotonic_increasing or not ts.is_unique:
        raise ValueError(f"{source_name}: timestamps must be strictly increasing.")
    if len(ts) > 1:
        deltas = np.diff(ts.to_numpy(dtype="datetime64[ns]")).astype("timedelta64[m]").astype(np.int64)
        if not np.all(deltas == 30):
            raise ValueError(f"{source_name}: timestamps must be exactly 30 minutes apart.")


def build_source_boundaries(sources: list[SourceSpec]) -> tuple[dict[str, SourceBoundary], np.ndarray]:
    boundaries: dict[str, SourceBoundary] = {}
    all_timestamps: list[np.ndarray] = []
    cursor = 0
    for source in sources:
        timestamps = source_timestamps(source)
        start_idx = cursor
        end_idx = cursor + len(timestamps) - 1
        boundaries[source.name] = SourceBoundary(
            name=source.name,
            start_idx=start_idx,
            end_idx=end_idx,
            start_timestamp=str(pd.Timestamp(timestamps[0])),
            end_timestamp=str(pd.Timestamp(timestamps[-1])),
        )
        all_timestamps.append(timestamps)
        cursor = end_idx + 1
    return boundaries, np.concatenate(all_timestamps)


def add_cyclic_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    hour_float = out["hour"].astype("float32") + out["minute"].astype("float32") / 60.0
    out["hour_sin"] = np.sin(2 * np.pi * hour_float / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour_float / 24.0)
    out["day_of_week_sin"] = np.sin(2 * np.pi * out["day_of_week"].astype("float32") / 7.0)
    out["day_of_week_cos"] = np.cos(2 * np.pi * out["day_of_week"].astype("float32") / 7.0)
    out["month_sin"] = np.sin(2 * np.pi * (out["month"].astype("float32") - 1.0) / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * (out["month"].astype("float32") - 1.0) / 12.0)
    return out


def build_time_frame(source: SourceSpec, timestamps: np.ndarray) -> pd.DataFrame:
    weather_path = source.path / "weather_30min.parquet"
    if not weather_path.exists():
        raise FileNotFoundError(f"Missing weather_30min.parquet for source {source.name}: {weather_path}")
    weather = pd.read_parquet(weather_path)
    missing = set(WEATHER_REQUIRED_COLUMNS) - set(weather.columns)
    if missing:
        raise ValueError(f"{source.name}: weather file missing columns: {sorted(missing)}")
    weather["timestamp"] = pd.to_datetime(weather["timestamp"])
    expected = pd.DataFrame({"timestamp": pd.DatetimeIndex(timestamps)})
    df = expected.merge(weather[WEATHER_REQUIRED_COLUMNS], on="timestamp", how="left")
    if df[WEATHER_REQUIRED_COLUMNS[1:]].isna().any().any():
        raise ValueError(f"{source.name}: weather_30min does not cover every source timestamp.")

    ts = df["timestamp"]
    df["hour"] = ts.dt.hour.astype("int8")
    df["minute"] = ts.dt.minute.astype("int8")
    df["day_of_week"] = ts.dt.dayofweek.astype("int8")
    df["month"] = ts.dt.month.astype("int8")
    df["is_weekend"] = ts.dt.dayofweek.isin([5, 6]).astype("int8")
    df["is_holiday"] = infer_holiday_flags_from_panel(source, timestamps)
    return add_cyclic_time_features(df)


def infer_holiday_flags_from_panel(source: SourceSpec, timestamps: np.ndarray) -> np.ndarray:
    panel_path = source.path / "station_time_panel.parquet"
    parquet_file = pq.ParquetFile(panel_path)
    pieces = []
    for row_group in range(parquet_file.num_row_groups):
        df = parquet_file.read_row_group(row_group, columns=["timestamp", "is_holiday"]).to_pandas()
        pieces.append(df.drop_duplicates("timestamp"))
    holidays = pd.concat(pieces, ignore_index=True).drop_duplicates("timestamp")
    holidays["timestamp"] = pd.to_datetime(holidays["timestamp"])
    aligned = pd.DataFrame({"timestamp": pd.DatetimeIndex(timestamps)}).merge(
        holidays,
        on="timestamp",
        how="left",
    )
    if aligned["is_holiday"].isna().any():
        raise ValueError(f"{source.name}: could not infer is_holiday for every timestamp from panel.")
    return aligned["is_holiday"].astype("int8").to_numpy()


def initialize_time_features(
    dynamic: np.memmap,
    source_slice: slice,
    time_frame: pd.DataFrame,
) -> None:
    for column in [
        "temperature",
        "wind_speed",
        "rainfall",
        "humidity",
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "month_sin",
        "month_cos",
        "is_weekend",
        "is_holiday",
    ]:
        feature_idx = DYNAMIC_FEATURE_COLUMNS.index(column)
        values = time_frame[column].to_numpy(dtype="float32")
        dynamic[source_slice, :, feature_idx] = values[:, None]


def populate_panel_features(
    dynamic: np.memmap,
    source: SourceSpec,
    boundary: SourceBoundary,
    timestamps: np.ndarray,
    station_numbers: np.ndarray,
) -> dict[str, Any]:
    panel_path = source.path / "station_time_panel.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(f"Missing panel for source {source.name}: {panel_path}")

    parquet_file = pq.ParquetFile(panel_path)
    station_index = pd.Series(np.arange(len(station_numbers), dtype=np.int64), index=station_numbers)
    timestamp_index = pd.Index(timestamps)
    observed_stations: set[int] = set()
    assigned_rows = 0

    for row_group in range(parquet_file.num_row_groups):
        logging.info("Reading %s panel row group %s/%s", source.name, row_group + 1, parquet_file.num_row_groups)
        df = parquet_file.read_row_group(row_group, columns=PANEL_REQUIRED_COLUMNS).to_pandas()
        missing = set(PANEL_REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"{source.name}: panel missing columns: {sorted(missing)}")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["station_number"] = df["station_number"].astype(np.int64)
        unknown = set(df["station_number"].unique().tolist()) - set(station_numbers.tolist())
        if unknown:
            raise ValueError(f"{source.name}: panel contains stations not in shared station order: {sorted(unknown)[:10]}")

        time_idx = timestamp_index.get_indexer(df["timestamp"].to_numpy(dtype="datetime64[ns]"))
        if (time_idx < 0).any():
            raise ValueError(f"{source.name}: panel contains timestamps outside source timestamp grid.")
        global_time_idx = time_idx + boundary.start_idx
        station_idx = station_index.loc[df["station_number"].to_numpy(dtype=np.int64)].to_numpy()
        observed_stations.update(df["station_number"].unique().astype(np.int64).tolist())

        values = df[
            [
                "rental_count",
                "return_count",
                "net_demand",
                "avg_duration_min",
                "avg_distance_m",
            ]
        ].to_numpy(dtype="float32", copy=False)
        dynamic[global_time_idx, station_idx, 0:5] = values
        assigned_rows += len(df)

    missing_stations = sorted(set(station_numbers.tolist()) - observed_stations)
    return {
        "panel_rows": int(parquet_file.metadata.num_rows),
        "assigned_rows": int(assigned_rows),
        "observed_station_count": int(len(observed_stations)),
        "missing_station_count": int(len(missing_stations)),
        "missing_station_numbers": [int(x) for x in missing_stations[:50]],
        "reindexed_missing_stations_with_zero_demand": bool(missing_stations),
    }


def build_dynamic_and_targets(
    output_dir: Path,
    sources: list[SourceSpec],
    boundaries: dict[str, SourceBoundary],
    timestamps_all: np.ndarray,
    station_numbers: np.ndarray,
    target_columns: list[str],
) -> tuple[np.memmap, np.memmap, np.memmap, dict[str, Any]]:
    T = len(timestamps_all)
    S = len(station_numbers)
    F = len(DYNAMIC_FEATURE_COLUMNS)
    target_dim = len(target_columns)
    dynamic = np.lib.format.open_memmap(
        output_dir / "dynamic_features.npy",
        mode="w+",
        dtype="float32",
        shape=(T, S, F),
    )
    targets_raw = np.lib.format.open_memmap(
        output_dir / "targets_raw.npy",
        mode="w+",
        dtype="float32",
        shape=(T, S, target_dim),
    )
    targets = np.lib.format.open_memmap(
        output_dir / "targets.npy",
        mode="w+",
        dtype="float32",
        shape=(T, S, target_dim),
    )
    dynamic[:] = 0.0
    source_reports: dict[str, Any] = {}

    for source in sources:
        boundary = boundaries[source.name]
        local_timestamps = timestamps_all[boundary.start_idx : boundary.end_idx + 1]
        source_slice = slice(boundary.start_idx, boundary.end_idx + 1)
        time_frame = build_time_frame(source, local_timestamps)
        initialize_time_features(dynamic, source_slice, time_frame)
        source_reports[source.name] = populate_panel_features(
            dynamic,
            source,
            boundary,
            local_timestamps,
            station_numbers,
        )

    for idx, column in enumerate(target_columns):
        targets_raw[:, :, idx] = dynamic[:, :, DYNAMIC_FEATURE_COLUMNS.index(column)]
    dynamic.flush()
    targets_raw.flush()
    return dynamic, targets, targets_raw, source_reports


def parse_start(value: str) -> pd.Timestamp:
    return pd.Timestamp(value)


def parse_end_exclusive(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if len(str(value)) <= 10:
        return ts + pd.Timedelta(days=1)
    return ts + pd.Timedelta(nanoseconds=1)


def make_sample_index_for_range(
    source_name: str,
    start: str,
    end: str,
    timestamps_all: np.ndarray,
    boundary: SourceBoundary,
    station_count: int,
    window_offsets: np.ndarray,
) -> np.ndarray:
    ts = pd.DatetimeIndex(timestamps_all[boundary.start_idx : boundary.end_idx + 1])
    start_ts = parse_start(start)
    end_exclusive = parse_end_exclusive(end)
    local_candidates = np.where((ts >= start_ts) & (ts < end_exclusive))[0]
    if len(local_candidates) == 0:
        raise ValueError(f"Split range selected no timestamps for source {source_name}: {start} to {end}")

    valid_local = local_candidates[
        (local_candidates + int(np.min(window_offsets)) >= 0)
        & (local_candidates + int(np.max(window_offsets)) < boundary.length)
    ]
    if len(valid_local) == 0:
        return np.empty((0, 2), dtype=np.int32)

    target_time_idx = valid_local.astype(np.int64) + boundary.start_idx
    repeated_times = np.repeat(target_time_idx, station_count)
    tiled_stations = np.tile(np.arange(station_count, dtype=np.int64), len(target_time_idx))
    pairs = np.column_stack([repeated_times, tiled_stations])
    if pairs.max(initial=0) > np.iinfo(np.int32).max:
        raise ValueError("Sample indices exceed int32 range.")
    return pairs.astype(np.int32, copy=False)


def build_sample_indices(
    output_dir: Path,
    config: dict[str, Any],
    boundaries: dict[str, SourceBoundary],
    timestamps_all: np.ndarray,
    station_count: int,
    window_offsets: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    split_config = config.get("splits")
    if not isinstance(split_config, dict) or not split_config:
        raise ValueError("Config must define a non-empty splits mapping.")

    split_arrays: dict[str, np.ndarray] = {}
    split_meta: dict[str, Any] = {}
    for split_name, ranges in split_config.items():
        if not isinstance(ranges, list) or not ranges:
            raise ValueError(f"Split {split_name} must contain one or more ranges.")
        pieces = []
        range_meta = []
        for item in ranges:
            source_name = str(item["source"])
            if source_name not in boundaries:
                raise ValueError(f"Split {split_name} references unknown source: {source_name}")
            pairs = make_sample_index_for_range(
                source_name,
                str(item["start"]),
                str(item["end"]),
                timestamps_all,
                boundaries[source_name],
                station_count,
                window_offsets,
            )
            pieces.append(pairs)
            valid_targets = int(len(np.unique(pairs[:, 0])) if len(pairs) else 0)
            range_meta.append(
                {
                    "source": source_name,
                    "start": str(item["start"]),
                    "end": str(item["end"]),
                    "valid_target_timestamp_count": valid_targets,
                    "sample_count": int(len(pairs)),
                }
            )
        sample_index = np.concatenate(pieces, axis=0) if pieces else np.empty((0, 2), dtype=np.int32)
        if len(sample_index) == 0:
            raise ValueError(f"Split {split_name} has no valid samples after window-boundary filtering.")
        filename = f"sample_index_{split_name}.npy"
        np.save(output_dir / filename, sample_index)
        split_arrays[split_name] = sample_index
        split_meta[split_name] = {
            "ranges": range_meta,
            "sample_index_file": filename,
            "sample_count": int(len(sample_index)),
            "valid_target_timestamp_count": int(len(np.unique(sample_index[:, 0]))),
        }
    return split_arrays, split_meta


def streaming_mean_std(
    arrays: list[tuple[np.ndarray, list[int] | np.ndarray, int]],
    transform: str,
    chunk_size: int = 512,
) -> tuple[float, float]:
    count = 0
    total = 0.0
    total_sq = 0.0
    for array, time_indices, feature_idx in arrays:
        indices = np.asarray(time_indices, dtype=np.int64)
        for start in range(0, len(indices), chunk_size):
            idx = indices[start : start + chunk_size]
            values = array[idx, :, feature_idx].astype("float64", copy=False)
            transformed = apply_transform(values, transform)
            total += float(transformed.sum(dtype=np.float64))
            total_sq += float(np.square(transformed, dtype=np.float64).sum(dtype=np.float64))
            count += int(transformed.size)
    if count == 0:
        raise ValueError("Cannot fit scaler on zero values.")
    mean = total / count
    variance = max(total_sq / count - mean * mean, 0.0)
    std = float(np.sqrt(variance))
    if std == 0.0:
        std = 1.0
    return float(mean), std


def fit_scalers(
    dynamic: np.ndarray,
    targets_raw: np.ndarray,
    split_arrays: dict[str, np.ndarray],
    station_metadata: pd.DataFrame,
    target_columns: list[str],
    window_offsets: np.ndarray,
    train_split_name: str,
) -> dict[str, Any]:
    if train_split_name not in split_arrays:
        raise ValueError(f"train_split_name={train_split_name!r} is not in splits.")
    train_pairs = split_arrays[train_split_name]
    train_target_times = np.unique(train_pairs[:, 0].astype(np.int64))
    train_input_times = np.unique(train_target_times[:, None] + window_offsets.astype(np.int64)[None, :])

    count_fit_arrays: list[tuple[np.ndarray, np.ndarray, int]] = []
    for column in COUNT_COLUMNS:
        count_fit_arrays.append((dynamic, train_input_times, DYNAMIC_FEATURE_COLUMNS.index(column)))
    for column in target_columns:
        if column in COUNT_COLUMNS:
            count_fit_arrays.append((targets_raw, train_target_times, target_columns.index(column)))
    count_mean, count_std = streaming_mean_std(count_fit_arrays, "log1p")

    scalers: dict[str, Any] = {
        "fit": {
            "train_split_name": train_split_name,
            "train_target_timestamp_count": int(len(train_target_times)),
            "train_input_timestamp_count": int(len(train_input_times)),
        },
        "count_scaler": {
            "columns": COUNT_COLUMNS,
            "transform": "log1p",
            "mean": count_mean,
            "std": count_std,
        },
        "net_demand_scaler": {},
        "dynamic_continuous": {},
        "static_numeric": {},
    }

    net_mean, net_std = streaming_mean_std(
        [(dynamic, train_input_times, DYNAMIC_FEATURE_COLUMNS.index("net_demand"))],
        "signed_log1p",
    )
    scalers["net_demand_scaler"] = {"transform": "signed_log1p", "mean": net_mean, "std": net_std}

    for column in CONTINUOUS_DYNAMIC_COLUMNS:
        mean, std = streaming_mean_std(
            [(dynamic, train_input_times, DYNAMIC_FEATURE_COLUMNS.index(column))],
            "none",
        )
        scalers["dynamic_continuous"][column] = {"transform": "none", "mean": mean, "std": std}

    for column in STATIC_NUMERIC_COLUMNS:
        values = station_metadata[column].to_numpy(dtype="float64")
        mean = float(values.mean())
        std = float(values.std())
        if std == 0.0:
            std = 1.0
        scalers["static_numeric"][column] = {"transform": "none", "mean": mean, "std": std}
    return scalers


def apply_scaling(
    dynamic: np.memmap,
    targets: np.memmap,
    targets_raw: np.ndarray,
    scalers: dict[str, Any],
    target_columns: list[str],
    chunk_size: int,
) -> None:
    count_scaler = scalers["count_scaler"]
    count_mean = float(count_scaler["mean"])
    count_std = float(count_scaler["std"])
    for column in COUNT_COLUMNS:
        feature_idx = DYNAMIC_FEATURE_COLUMNS.index(column)
        for start in range(0, dynamic.shape[0], chunk_size):
            end = min(start + chunk_size, dynamic.shape[0])
            transformed = apply_transform(dynamic[start:end, :, feature_idx], "log1p")
            dynamic[start:end, :, feature_idx] = standardize(transformed, count_mean, count_std).astype("float32")

    net_scaler = scalers["net_demand_scaler"]
    for start in range(0, dynamic.shape[0], chunk_size):
        end = min(start + chunk_size, dynamic.shape[0])
        transformed = apply_transform(dynamic[start:end, :, DYNAMIC_FEATURE_COLUMNS.index("net_demand")], "signed_log1p")
        dynamic[start:end, :, DYNAMIC_FEATURE_COLUMNS.index("net_demand")] = standardize(
            transformed,
            float(net_scaler["mean"]),
            float(net_scaler["std"]),
        ).astype("float32")

    for column, scaler in scalers["dynamic_continuous"].items():
        feature_idx = DYNAMIC_FEATURE_COLUMNS.index(column)
        for start in range(0, dynamic.shape[0], chunk_size):
            end = min(start + chunk_size, dynamic.shape[0])
            dynamic[start:end, :, feature_idx] = standardize(
                dynamic[start:end, :, feature_idx],
                float(scaler["mean"]),
                float(scaler["std"]),
            ).astype("float32")

    for idx, column in enumerate(target_columns):
        if column not in COUNT_COLUMNS:
            raise ValueError(f"Only count targets are currently supported for scaling, got {column}")
        for start in range(0, targets.shape[0], chunk_size):
            end = min(start + chunk_size, targets.shape[0])
            transformed = apply_transform(targets_raw[start:end, :, idx], "log1p")
            targets[start:end, :, idx] = standardize(transformed, count_mean, count_std).astype("float32")
    dynamic.flush()
    targets.flush()


def canonical_operation_type(value: object) -> str:
    normalized = str(value).replace(" ", "").upper()
    if "QR" in normalized:
        return "QR"
    if "LCD" in normalized:
        return "LCD"
    return normalized


def build_static_outputs(
    output_dir: Path,
    station_metadata: pd.DataFrame,
    station_numbers: np.ndarray,
    scalers: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int], dict[str, int], dict[str, int]]:
    static_numeric = np.empty((len(station_metadata), len(STATIC_NUMERIC_COLUMNS)), dtype="float32")
    for idx, column in enumerate(STATIC_NUMERIC_COLUMNS):
        scaler = scalers["static_numeric"][column]
        static_numeric[:, idx] = standardize(
            station_metadata[column].to_numpy(dtype="float32"),
            float(scaler["mean"]),
            float(scaler["std"]),
        ).astype("float32")

    districts = station_metadata["district"].astype(str)
    district_vocab = {value: idx for idx, value in enumerate(sorted(districts.unique().tolist()))}
    district_ids = districts.map(district_vocab).to_numpy(dtype=np.int64)

    operation_types = station_metadata["operation_type"].map(canonical_operation_type).astype(str)
    operation_type_vocab = {value: idx for idx, value in enumerate(sorted(operation_types.unique().tolist()))}
    operation_type_ids = operation_types.map(operation_type_vocab).to_numpy(dtype=np.int64)

    station_index_map = {str(int(number)): int(idx) for idx, number in enumerate(station_numbers)}

    np.save(output_dir / "static_numeric.npy", static_numeric.astype("float32"))
    np.save(output_dir / "district_ids.npy", district_ids.astype(np.int64))
    np.save(output_dir / "operation_type_ids.npy", operation_type_ids.astype(np.int64))
    np.save(output_dir / "station_numbers.npy", station_numbers.astype(np.int64))
    return static_numeric, district_ids, operation_type_ids, station_index_map, district_vocab, operation_type_vocab


def validate_sample_indices(
    split_arrays: dict[str, np.ndarray],
    boundaries: dict[str, SourceBoundary],
    T: int,
    S: int,
    window_offsets: np.ndarray,
) -> None:
    boundary_ranges = [(boundary.start_idx, boundary.end_idx) for boundary in boundaries.values()]
    for split_name, sample_index in split_arrays.items():
        if sample_index.ndim != 2 or sample_index.shape[1] != 2:
            raise ValueError(f"{split_name}: sample index must have shape (N, 2).")
        if sample_index.dtype != np.int32:
            raise ValueError(f"{split_name}: sample index must use int32.")
        if len(sample_index) == 0:
            raise ValueError(f"{split_name}: sample index is empty.")
        if sample_index[:, 0].min() < 0 or sample_index[:, 0].max() >= T:
            raise ValueError(f"{split_name}: target time index out of bounds.")
        if sample_index[:, 1].min() < 0 or sample_index[:, 1].max() >= S:
            raise ValueError(f"{split_name}: station index out of bounds.")

        target_times = np.unique(sample_index[:, 0].astype(np.int64))
        for target_time in target_times:
            matching = [bounds for bounds in boundary_ranges if bounds[0] <= target_time <= bounds[1]]
            if len(matching) != 1:
                raise ValueError(f"{split_name}: target time {target_time} does not map to exactly one source.")
            start_idx, end_idx = matching[0]
            input_times = target_time + window_offsets.astype(np.int64)
            if input_times.min() < start_idx or input_times.max() > end_idx:
                raise ValueError(f"{split_name}: sample at target {target_time} crosses a source boundary.")


def validate_outputs(
    output_dir: Path,
    dynamic: np.ndarray,
    targets: np.ndarray,
    targets_raw: np.ndarray,
    station_numbers: np.ndarray,
    shared_station_numbers: np.ndarray,
    split_arrays: dict[str, np.ndarray],
    boundaries: dict[str, SourceBoundary],
    window_offsets: np.ndarray,
    horizon: int,
) -> dict[str, int]:
    T, S, F = dynamic.shape
    if F != len(DYNAMIC_FEATURE_COLUMNS):
        raise ValueError("dynamic feature dimension mismatch.")
    if targets.shape[:2] != (T, S) or targets_raw.shape != targets.shape:
        raise ValueError("targets and targets_raw must share shape (T, S, target_dim).")
    if not np.array_equal(station_numbers, shared_station_numbers):
        raise ValueError("Saved station_numbers do not match shared station order.")
    validate_window_offsets(window_offsets, horizon)
    validate_sample_indices(split_arrays, boundaries, T, S, window_offsets)

    missing = {
        "dynamic_features_nan_or_inf": count_nonfinite(dynamic),
        "targets_nan_or_inf": count_nonfinite(targets),
        "targets_raw_nan_or_inf": count_nonfinite(targets_raw),
    }
    if any(missing.values()):
        raise ValueError(f"Output tensors contain NaN or inf values: {missing}")

    for filename, dtype in [
        ("dynamic_features.npy", np.float32),
        ("targets.npy", np.float32),
        ("targets_raw.npy", np.float32),
        ("static_numeric.npy", np.float32),
    ]:
        arr = np.load(output_dir / filename, mmap_mode="r")
        if arr.dtype != dtype:
            raise ValueError(f"{filename} must be {dtype}, got {arr.dtype}")
    return missing


def count_nonfinite(array: np.ndarray, chunk_size: int = 512) -> int:
    total = 0
    for start in range(0, array.shape[0], chunk_size):
        end = min(start + chunk_size, array.shape[0])
        total += int((~np.isfinite(array[start:end])).sum())
    return total


def boundary_json(boundaries: dict[str, SourceBoundary]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "start_idx": boundary.start_idx,
            "end_idx": boundary.end_idx,
            "start_timestamp": boundary.start_timestamp,
            "end_timestamp": boundary.end_timestamp,
        }
        for name, boundary in boundaries.items()
    }


def save_metadata(
    output_dir: Path,
    config: dict[str, Any],
    sources: list[SourceSpec],
    boundaries: dict[str, SourceBoundary],
    timestamps_all: np.ndarray,
    station_numbers: np.ndarray,
    window_offsets: np.ndarray,
    target_columns: list[str],
    scalers: dict[str, Any],
    split_meta: dict[str, Any],
    source_reports: dict[str, Any],
    station_index_map: dict[str, int],
    district_vocab: dict[str, int],
    operation_type_vocab: dict[str, int],
    missing_summary: dict[str, int],
) -> None:
    source_boundaries = boundary_json(boundaries)
    write_json(output_dir / "source_boundaries.json", source_boundaries)
    write_json(output_dir / "station_index_map.json", station_index_map)
    write_json(output_dir / "district_vocab.json", district_vocab)
    write_json(output_dir / "operation_type_vocab.json", operation_type_vocab)
    write_json(output_dir / "scalers.json", scalers)
    write_json(output_dir / "splits.json", split_meta)

    feature_config = {
        "horizon": int(config.get("horizon", 1)),
        "window_offsets": window_offsets.astype(int).tolist(),
        "dynamic_feature_columns": DYNAMIC_FEATURE_COLUMNS,
        "target_columns": target_columns,
        "static_numeric_columns": [f"{name}_scaled" for name in STATIC_NUMERIC_COLUMNS],
        "categorical_static_columns": STATIC_CATEGORICAL_COLUMNS,
        "transformations": {
            "count_columns": COUNT_COLUMNS,
            "count_transform": "log1p",
            "net_demand_transform": "signed_log1p",
            "continuous_dynamic_transform": "standard",
            "static_numeric_transform": "standard",
        },
    }
    write_json(output_dir / "feature_config.json", feature_config)

    output_files = sorted(path.name for path in output_dir.iterdir() if path.is_file())
    summary = {
        "dataset_name": config.get("dataset_name", output_dir.name),
        "sources": [{"name": source.name, "path": str(source.path)} for source in sources],
        "source_boundaries": source_boundaries,
        "source_reports": source_reports,
        "T_total": int(len(timestamps_all)),
        "S": int(len(station_numbers)),
        "dynamic_feature_dim": len(DYNAMIC_FEATURE_COLUMNS),
        "target_dim": len(target_columns),
        "window_size": int(len(window_offsets)),
        "horizon": int(config.get("horizon", 1)),
        "samples_per_split": {name: int(meta["sample_count"]) for name, meta in split_meta.items()},
        "valid_target_timestamps_per_split": {
            name: int(meta["valid_target_timestamp_count"]) for name, meta in split_meta.items()
        },
        "split_ranges": {name: meta["ranges"] for name, meta in split_meta.items()},
        "missing_value_summary": missing_summary,
        "generated_output_files": output_files,
    }
    write_json(output_dir / "dataset_summary.json", summary)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    config = apply_cli_overrides(load_config(args.config), args)

    output_dir = resolve_output_dir(config)
    station_dir = Path(config.get("station_dir", "data/preprocessed/station"))
    sources = resolve_sources(config)
    horizon = int(config.get("horizon", 1))
    target_columns = [str(x) for x in config.get("target_columns", TARGET_COLUMNS_DEFAULT)]
    if target_columns != TARGET_COLUMNS_DEFAULT:
        raise ValueError("This baseline builder currently supports target_columns: rental_count, return_count.")

    station_metadata, station_numbers = load_station_inputs(station_dir)
    window_offsets = build_window_offsets(config.get("window", {}))
    validate_window_offsets(window_offsets, horizon)
    np.save(output_dir / "window_offsets.npy", window_offsets.astype(np.int32))

    boundaries, timestamps_all = build_source_boundaries(sources)
    np.save(output_dir / "timestamps.npy", timestamps_all)

    logging.info("Building tensors: T=%s, S=%s, F=%s", len(timestamps_all), len(station_numbers), len(DYNAMIC_FEATURE_COLUMNS))
    dynamic, targets, targets_raw, source_reports = build_dynamic_and_targets(
        output_dir,
        sources,
        boundaries,
        timestamps_all,
        station_numbers,
        target_columns,
    )

    split_arrays, split_meta = build_sample_indices(
        output_dir,
        config,
        boundaries,
        timestamps_all,
        len(station_numbers),
        window_offsets,
    )

    train_split_name = str(config.get("train_split_name", "train"))
    scalers = fit_scalers(
        dynamic,
        targets_raw,
        split_arrays,
        station_metadata,
        target_columns,
        window_offsets,
        train_split_name,
    )
    apply_scaling(
        dynamic,
        targets,
        targets_raw,
        scalers,
        target_columns,
        chunk_size=int(config.get("scaling_chunk_size", 512)),
    )

    static_numeric, district_ids, operation_type_ids, station_index_map, district_vocab, operation_type_vocab = build_static_outputs(
        output_dir,
        station_metadata,
        station_numbers,
        scalers,
    )
    missing_summary = validate_outputs(
        output_dir,
        dynamic,
        targets,
        targets_raw,
        np.load(output_dir / "station_numbers.npy"),
        station_numbers,
        split_arrays,
        boundaries,
        window_offsets,
        horizon,
    )
    if not np.isfinite(static_numeric).all():
        raise ValueError("static_numeric contains NaN or inf values.")
    if district_ids.shape != (len(station_numbers),) or operation_type_ids.shape != (len(station_numbers),):
        raise ValueError("Categorical station arrays have invalid shapes.")

    save_metadata(
        output_dir,
        config,
        sources,
        boundaries,
        timestamps_all,
        station_numbers,
        window_offsets,
        target_columns,
        scalers,
        split_meta,
        source_reports,
        station_index_map,
        district_vocab,
        operation_type_vocab,
        missing_summary,
    )
    logging.info("LSTM dataset complete: %s", output_dir)


if __name__ == "__main__":
    main()
