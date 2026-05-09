from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyYAML is required. Install it with: pip install pyyaml") from exc


RENTAL_FEATURE_COLUMNS = ["rental_count", "temperature", "wind_speed", "rainfall", "humidity"]
RETURN_FEATURE_COLUMNS = ["return_count", "temperature", "wind_speed", "rainfall", "humidity"]
TARGET_COLUMNS = ["rental_count", "return_count"]
WEATHER_COLUMNS = ["temperature", "wind_speed", "rainfall", "humidity"]
TARGET_TIME_FEATURE_COLUMNS = [
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "is_holiday",
]
PANEL_COLUMNS = [
    "timestamp",
    "station_number",
    "rental_count",
    "return_count",
    "temperature",
    "wind_speed",
    "rainfall",
    "humidity",
    "hour",
    "minute",
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday",
]
OUTPUT_FILES = [
    "dataset_summary.json",
    "feature_config.json",
    "base_data.json",
    "scalers.json",
    "source_boundaries.json",
    "timestamps.npy",
    "station_numbers.npy",
    "rental_features.npy",
    "return_features.npy",
    "targets.npy",
    "targets_raw.npy",
    "future_weather_features.npy",
    "target_time_features.npy",
    "static_numeric.npy",
    "district_ids.npy",
    "operation_type_ids.npy",
    "window_offsets.npy",
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


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compact TCT-GAT graph-snapshot arrays.")
    parser.add_argument("--config", type=Path, default=Path("configs/data/tct_gat/tct_gat1_ar.yaml"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_output_dir(output_dir: Path, overwrite: bool) -> None:
    existing = [output_dir / name for name in OUTPUT_FILES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "TCT-GAT dataset outputs already exist. Use --overwrite or choose a new output_dir. "
            f"Existing files: {[str(path) for path in existing[:8]]}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def resolve_sources(config: dict[str, Any]) -> list[SourceSpec]:
    sources = config.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Config must define a non-empty sources list.")
    seen: set[str] = set()
    resolved: list[SourceSpec] = []
    for item in sources:
        name = str(item["name"])
        if name in seen:
            raise ValueError(f"Duplicate source name: {name}")
        seen.add(name)
        resolved.append(SourceSpec(name=name, path=Path(item["path"])))
    return resolved


def build_window_offsets(config: dict[str, Any]) -> np.ndarray:
    window = config.get("window", {})
    offsets = [
        int(value)
        for value in (
            list(window.get("recent_offsets", []))
            + list(window.get("daily_offsets", []))
            + list(window.get("weekly_offsets", []))
        )
    ]
    if not offsets:
        raise ValueError("At least one window offset is required.")
    if any(offset >= 0 for offset in offsets):
        raise ValueError("TCT-GAT window offsets must be negative.")
    return np.asarray(sorted(set(offsets)), dtype=np.int32)


def source_timestamps(source: SourceSpec) -> np.ndarray:
    summary_path = source.path / "preprocessing_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing preprocessing summary for {source.name}: {summary_path}")
    summary = read_json(summary_path)
    timestamps = pd.date_range(summary["start_timestamp"], summary["end_timestamp"], freq="30min")
    expected = int(summary["num_timestamps"])
    if len(timestamps) != expected:
        raise ValueError(f"{source.name}: expected {expected} timestamps, got {len(timestamps)}")
    return timestamps.to_numpy(dtype="datetime64[ns]")


def build_source_boundaries(sources: list[SourceSpec]) -> tuple[dict[str, SourceBoundary], np.ndarray]:
    boundaries: dict[str, SourceBoundary] = {}
    pieces: list[np.ndarray] = []
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
        pieces.append(timestamps)
        cursor = end_idx + 1
    return boundaries, np.concatenate(pieces)


def load_station_metadata(station_dir: Path) -> pd.DataFrame:
    path = station_dir / "station_metadata_clean.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing station metadata: {path}")
    metadata = pd.read_parquet(path)
    required = {"station_number", "latitude", "longitude", "dock_count_raw", "district", "operation_type"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Station metadata missing columns: {sorted(missing)}")
    metadata = metadata.sort_values("station_number").drop_duplicates("station_number", keep="first").reset_index(drop=True)
    metadata["station_number"] = metadata["station_number"].astype(np.int64)
    return metadata


def make_vocab(values: pd.Series) -> dict[str, int]:
    labels = sorted(str(value) for value in values.astype("string").fillna("unknown").unique())
    return {label: idx for idx, label in enumerate(labels)}


def normalize_operation_label(value: object) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    text = str(value).upper()
    if "LCD" in text:
        return "LCD"
    if "QR" in text:
        return "QR"
    return "unknown"


def fit_standard_scaler(values: np.ndarray, transform: str | None = None) -> dict[str, float | str]:
    arr = np.asarray(values, dtype=np.float64)
    if transform == "log1p":
        arr = np.log1p(np.clip(arr, 0.0, None))
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if std < 1.0e-6:
        std = 1.0
    return {"mean": mean, "std": std, "transform": transform or "identity"}


def apply_scaler(values: np.ndarray, scaler: dict[str, Any]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if scaler.get("transform") == "log1p":
        arr = np.log1p(np.clip(arr, 0.0, None)).astype(np.float32)
    return ((arr - float(scaler["mean"])) / float(scaler["std"])).astype(np.float32)


def train_time_indices(boundaries: dict[str, SourceBoundary], config: dict[str, Any]) -> np.ndarray:
    train_name = str(config.get("train_split_name", "train"))
    split_items = config["splits"][train_name]
    selected: list[np.ndarray] = []
    for item in split_items:
        boundary = boundaries[str(item["source"])]
        timestamps = pd.date_range(boundary.start_timestamp, boundary.end_timestamp, freq="30min")
        start = pd.Timestamp(item["start"])
        end = pd.Timestamp(item["end"]) + pd.Timedelta(days=1)
        local = np.flatnonzero((timestamps >= start) & (timestamps < end)).astype(np.int64)
        selected.append(local + boundary.start_idx)
    if not selected:
        raise ValueError("No train timestamps found.")
    return np.concatenate(selected)


def fit_scalers(
    config: dict[str, Any],
    station_metadata: pd.DataFrame,
    sources: list[SourceSpec],
    boundaries: dict[str, SourceBoundary],
    station_to_idx: pd.Series,
) -> dict[str, Any]:
    logging.info("Fitting TCT-GAT scalers on train split only")
    train_idx = set(train_time_indices(boundaries, config).tolist())
    rental_values: list[np.ndarray] = []
    return_values: list[np.ndarray] = []
    weather_values: list[np.ndarray] = []
    for source in sources:
        boundary = boundaries[source.name]
        source_train_global = [idx for idx in range(boundary.start_idx, boundary.end_idx + 1) if idx in train_idx]
        if not source_train_global:
            continue
        source_train_local = np.asarray(source_train_global, dtype=np.int64) - boundary.start_idx
        timestamps = pd.DatetimeIndex(source_timestamps(source))[source_train_local]
        parquet_file = pq.ParquetFile(source.path / "station_time_panel.parquet")
        for row_group in range(parquet_file.num_row_groups):
            df = parquet_file.read_row_group(row_group, columns=["timestamp", "station_number", *TARGET_COLUMNS, *WEATHER_COLUMNS]).to_pandas()
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df[df["timestamp"].isin(timestamps)]
            if df.empty:
                continue
            df["station_number"] = df["station_number"].astype(np.int64)
            df = df[df["station_number"].isin(station_to_idx.index)]
            rental_values.append(pd.to_numeric(df["rental_count"], errors="coerce").fillna(0).to_numpy(dtype=np.float32))
            return_values.append(pd.to_numeric(df["return_count"], errors="coerce").fillna(0).to_numpy(dtype=np.float32))
            weather_values.append(df[WEATHER_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=np.float32))

    if not rental_values:
        raise ValueError("No train rows found for scaler fitting.")
    scaling = config.get("scaling", {})
    count_transform = str(scaling.get("count_transform", "log1p"))
    scalers = {
        "target": {
            "rental_count": fit_standard_scaler(np.concatenate(rental_values), count_transform),
            "return_count": fit_standard_scaler(np.concatenate(return_values), count_transform),
        },
        "weather": {},
        "static_numeric": {},
    }
    weather = np.concatenate(weather_values, axis=0)
    for idx, column in enumerate(WEATHER_COLUMNS):
        scalers["weather"][column] = fit_standard_scaler(weather[:, idx])
    for column in config.get("static_numeric_columns", ["latitude", "longitude", "dock_count_raw"]):
        scalers["static_numeric"][column] = fit_standard_scaler(pd.to_numeric(station_metadata[column], errors="coerce").fillna(0).to_numpy())
    return scalers


def create_arrays(output_dir: Path, total_timestamps: int, num_stations: int, num_offsets: int) -> dict[str, np.memmap]:
    return {
        "rental_features": np.lib.format.open_memmap(output_dir / "rental_features.npy", mode="w+", dtype="float32", shape=(total_timestamps, num_stations, 5)),
        "return_features": np.lib.format.open_memmap(output_dir / "return_features.npy", mode="w+", dtype="float32", shape=(total_timestamps, num_stations, 5)),
        "targets": np.lib.format.open_memmap(output_dir / "targets.npy", mode="w+", dtype="float32", shape=(total_timestamps, num_stations, 2)),
        "targets_raw": np.lib.format.open_memmap(output_dir / "targets_raw.npy", mode="w+", dtype="float32", shape=(total_timestamps, num_stations, 2)),
        "future_weather_features": np.lib.format.open_memmap(output_dir / "future_weather_features.npy", mode="w+", dtype="float32", shape=(total_timestamps, 4)),
        "target_time_features": np.lib.format.open_memmap(output_dir / "target_time_features.npy", mode="w+", dtype="float32", shape=(total_timestamps, 8)),
    }


def compute_time_features(timestamps: pd.Series, is_holiday: pd.Series) -> np.ndarray:
    hour_float = timestamps.dt.hour.astype("float32") + timestamps.dt.minute.astype("float32") / 60.0
    out = np.column_stack(
        [
            np.sin(2.0 * np.pi * hour_float / 24.0),
            np.cos(2.0 * np.pi * hour_float / 24.0),
            np.sin(2.0 * np.pi * timestamps.dt.dayofweek.astype("float32") / 7.0),
            np.cos(2.0 * np.pi * timestamps.dt.dayofweek.astype("float32") / 7.0),
            np.sin(2.0 * np.pi * (timestamps.dt.month.astype("float32") - 1.0) / 12.0),
            np.cos(2.0 * np.pi * (timestamps.dt.month.astype("float32") - 1.0) / 12.0),
            timestamps.dt.dayofweek.isin([5, 6]).astype("float32"),
            is_holiday.astype("float32"),
        ]
    )
    return out.astype(np.float32)


def populate_source_arrays(
    source: SourceSpec,
    boundary: SourceBoundary,
    arrays: dict[str, np.memmap],
    station_to_idx: pd.Series,
    scalers: dict[str, Any],
) -> None:
    logging.info("Populating TCT-GAT arrays from %s", source.path)
    timestamps = pd.DatetimeIndex(source_timestamps(source))
    time_index = pd.Series(np.arange(len(timestamps), dtype=np.int64), index=timestamps)
    parquet_file = pq.ParquetFile(source.path / "station_time_panel.parquet")
    weather_seen = np.zeros((len(timestamps), len(WEATHER_COLUMNS)), dtype=np.float32)
    weather_count = np.zeros(len(timestamps), dtype=np.int32)
    holiday_seen = np.zeros(len(timestamps), dtype=np.float32)
    holiday_count = np.zeros(len(timestamps), dtype=np.int32)

    for row_group in range(parquet_file.num_row_groups):
        logging.info("Reading %s row group %s/%s", source.name, row_group + 1, parquet_file.num_row_groups)
        df = parquet_file.read_row_group(row_group, columns=PANEL_COLUMNS).to_pandas()
        missing = set(PANEL_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"{source.name}: panel missing columns: {sorted(missing)}")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["station_number"] = df["station_number"].astype(np.int64)
        known = df["station_number"].isin(station_to_idx.index)
        if not known.all():
            raise ValueError(f"{source.name}: panel contains stations missing from station metadata.")
        t_local = time_index.loc[df["timestamp"]].to_numpy(dtype=np.int64)
        t_global = t_local + boundary.start_idx
        s_idx = station_to_idx.loc[df["station_number"]].to_numpy(dtype=np.int64)

        raw_counts = df[TARGET_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        weather_raw = df[WEATHER_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        rental_scaled = apply_scaler(raw_counts[:, 0], scalers["target"]["rental_count"])
        return_scaled = apply_scaler(raw_counts[:, 1], scalers["target"]["return_count"])
        weather_scaled = np.column_stack(
            [apply_scaler(weather_raw[:, idx], scalers["weather"][column]) for idx, column in enumerate(WEATHER_COLUMNS)]
        ).astype(np.float32)

        arrays["rental_features"][t_global, s_idx, 0] = rental_scaled
        arrays["rental_features"][t_global, s_idx, 1:] = weather_scaled
        arrays["return_features"][t_global, s_idx, 0] = return_scaled
        arrays["return_features"][t_global, s_idx, 1:] = weather_scaled
        arrays["targets"][t_global, s_idx, 0] = rental_scaled
        arrays["targets"][t_global, s_idx, 1] = return_scaled
        arrays["targets_raw"][t_global, s_idx, :] = raw_counts

        np.add.at(weather_seen, t_local, weather_scaled)
        np.add.at(weather_count, t_local, 1)
        holiday = pd.to_numeric(df["is_holiday"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        np.add.at(holiday_seen, t_local, holiday)
        np.add.at(holiday_count, t_local, 1)

    if np.any(weather_count == 0):
        raise ValueError(f"{source.name}: panel did not populate weather for every timestamp.")
    avg_weather = weather_seen / weather_count[:, None]
    arrays["future_weather_features"][boundary.start_idx : boundary.end_idx + 1, :] = avg_weather
    holiday = pd.Series((holiday_seen / np.maximum(holiday_count, 1)) > 0.5)
    time_features = compute_time_features(pd.Series(timestamps), holiday)
    arrays["target_time_features"][boundary.start_idx : boundary.end_idx + 1, :] = time_features


def build_sample_indices(
    config: dict[str, Any],
    boundaries: dict[str, SourceBoundary],
    window_offsets: np.ndarray,
    output_dir: Path,
) -> dict[str, int]:
    samples_per_split: dict[str, int] = {}
    min_offset = int(window_offsets.min())
    for split_name, split_items in config["splits"].items():
        pieces: list[np.ndarray] = []
        for item in split_items:
            source_name = str(item["source"])
            boundary = boundaries[source_name]
            timestamps = pd.date_range(boundary.start_timestamp, boundary.end_timestamp, freq="30min")
            start = pd.Timestamp(item["start"])
            end = pd.Timestamp(item["end"]) + pd.Timedelta(days=1)
            local = np.flatnonzero((timestamps >= start) & (timestamps < end)).astype(np.int64)
            if local.size:
                valid = local + min_offset >= 0
                pieces.append(local[valid] + boundary.start_idx)
        sample_index = np.concatenate(pieces).astype(np.int64) if pieces else np.empty(0, dtype=np.int64)
        np.save(output_dir / f"sample_time_index_{split_name}.npy", sample_index)
        samples_per_split[split_name] = int(sample_index.size)
    return samples_per_split


def serialize_splits(config: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    serialized: dict[str, list[dict[str, str]]] = {}
    for split_name, items in config["splits"].items():
        serialized[split_name] = [
            {
                "source": str(item["source"]),
                "start": str(item["start"]),
                "end": str(item["end"]),
            }
            for item in items
        ]
    return serialized


def validate_graph_dir(graph_dir: Path, num_stations: int) -> None:
    required = [
        "neighbor_index_rr.npy",
        "neighbor_index_dd.npy",
        "neighbor_index_rd.npy",
        "neighbor_index_dr.npy",
        "edge_attr_rr.npy",
        "edge_attr_dd.npy",
        "edge_attr_rd.npy",
        "edge_attr_dr.npy",
        "edge_feature_columns.json",
        "graph_summary.json",
    ]
    missing = [name for name in required if not (graph_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Graph dir is missing required files: {missing}")
    summary = read_json(graph_dir / "graph_summary.json")
    if int(summary["num_stations"]) != num_stations:
        raise ValueError(f"Graph station count {summary['num_stations']} does not match dataset S={num_stations}.")


def build_dataset(config: dict[str, Any]) -> None:
    output_dir = Path(config.get("output_dir", "data/tct_gat_processed/tct_gat1_ar"))
    validate_output_dir(output_dir, bool(config.get("overwrite", False)))
    sources = resolve_sources(config)
    boundaries, timestamps = build_source_boundaries(sources)
    station_metadata = load_station_metadata(Path(config.get("station_dir", "data/preprocessed/station")))
    station_numbers = station_metadata["station_number"].to_numpy(dtype=np.int64)
    validate_graph_dir(Path(config["graph_dir"]), len(station_numbers))
    station_to_idx = pd.Series(np.arange(len(station_numbers), dtype=np.int64), index=station_numbers)
    window_offsets = build_window_offsets(config)

    district_vocab = make_vocab(station_metadata["district"])
    operation_labels = station_metadata["operation_type"].map(normalize_operation_label)
    operation_vocab = make_vocab(operation_labels)
    static_columns = list(config.get("static_numeric_columns", ["latitude", "longitude", "dock_count_raw"]))
    scalers = fit_scalers(config, station_metadata, sources, boundaries, station_to_idx)
    arrays = create_arrays(output_dir, len(timestamps), len(station_numbers), len(window_offsets))

    for source in sources:
        populate_source_arrays(source, boundaries[source.name], arrays, station_to_idx, scalers)

    static_raw = station_metadata[static_columns].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    static_scaled = np.column_stack(
        [apply_scaler(static_raw[:, idx], scalers["static_numeric"][column]) for idx, column in enumerate(static_columns)]
    ).astype(np.float32)
    district_ids = station_metadata["district"].astype("string").fillna("unknown").map(district_vocab).to_numpy(dtype=np.int64)
    operation_type_ids = operation_labels.astype("string").fillna("unknown").map(operation_vocab).to_numpy(dtype=np.int64)

    np.save(output_dir / "timestamps.npy", timestamps)
    np.save(output_dir / "station_numbers.npy", station_numbers)
    np.save(output_dir / "static_numeric.npy", static_scaled)
    np.save(output_dir / "district_ids.npy", district_ids)
    np.save(output_dir / "operation_type_ids.npy", operation_type_ids)
    np.save(output_dir / "window_offsets.npy", window_offsets)
    for array in arrays.values():
        array.flush()

    samples_per_split = build_sample_indices(config, boundaries, window_offsets, output_dir)
    feature_config = {
        "rental_feature_columns": RENTAL_FEATURE_COLUMNS,
        "return_feature_columns": RETURN_FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "future_weather_feature_columns": WEATHER_COLUMNS,
        "target_time_feature_columns": TARGET_TIME_FEATURE_COLUMNS,
        "static_numeric_columns": static_columns,
        "categorical_static_columns": list(config.get("categorical_static_columns", ["district", "operation_type"])),
        "window_offsets": window_offsets.astype(int).tolist(),
        "recent_offsets": [int(v) for v in config["window"]["recent_offsets"]],
        "daily_offsets": [int(v) for v in config["window"]["daily_offsets"]],
        "weekly_offsets": [int(v) for v in config["window"]["weekly_offsets"]],
        "district_vocab": district_vocab,
        "operation_type_vocab": operation_vocab,
    }
    base_data = {
        name: str(output_dir / f"{name}.npy")
        for name in [
            "timestamps",
            "station_numbers",
            "rental_features",
            "return_features",
            "targets",
            "targets_raw",
            "future_weather_features",
            "target_time_features",
            "static_numeric",
            "district_ids",
            "operation_type_ids",
            "window_offsets",
        ]
    }
    summary = {
        "dataset_name": config.get("dataset_name", "tct_gat1_ar"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "num_stations": int(len(station_numbers)),
        "num_timestamps": int(len(timestamps)),
        "S": int(len(station_numbers)),
        "T": int(len(timestamps)),
        "window_offsets": window_offsets.astype(int).tolist(),
        "num_window_offsets": int(len(window_offsets)),
        "splits": serialize_splits(config),
        "samples_per_split": samples_per_split,
        "feature_shapes": {
            "rental_features": list(arrays["rental_features"].shape),
            "return_features": list(arrays["return_features"].shape),
            "targets": list(arrays["targets"].shape),
            "targets_raw": list(arrays["targets_raw"].shape),
            "future_weather_features": list(arrays["future_weather_features"].shape),
            "target_time_features": list(arrays["target_time_features"].shape),
            "static_numeric": list(static_scaled.shape),
        },
        "target_columns": TARGET_COLUMNS,
        "rental_feature_columns": RENTAL_FEATURE_COLUMNS,
        "return_feature_columns": RETURN_FEATURE_COLUMNS,
        "future_weather_columns": WEATHER_COLUMNS,
        "target_time_feature_columns": TARGET_TIME_FEATURE_COLUMNS,
        "static_columns": static_columns,
        "graph_dir": str(config["graph_dir"]),
        "source_paths": {source.name: str(source.path) for source in sources},
        "scaling_summary": scalers,
    }
    write_json(output_dir / "feature_config.json", feature_config)
    write_json(output_dir / "base_data.json", base_data)
    write_json(output_dir / "scalers.json", scalers)
    write_json(output_dir / "source_boundaries.json", {name: vars(boundary) for name, boundary in boundaries.items()})
    write_json(output_dir / "dataset_summary.json", summary)
    logging.info("Wrote TCT-GAT dataset to %s", output_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    config = load_config(args.config)
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    if args.overwrite:
        config["overwrite"] = True
    build_dataset(config)


if __name__ == "__main__":
    main()
