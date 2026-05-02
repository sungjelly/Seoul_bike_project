from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.data.lstm_baseline.scaling import apply_transform, standardize


SEQUENCE_FEATURES = [
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

DYNAMIC_SCALED_FEATURES = [
    "rental_count",
    "return_count",
    "net_demand",
    "avg_duration_min",
    "avg_distance_m",
    "temperature",
    "wind_speed",
    "rainfall",
    "humidity",
]

DYNAMIC_UNSCALED_FEATURES = [
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "is_holiday",
]

TARGET_COLUMNS = ["rental_count", "return_count"]
STATIC_NUMERIC_FEATURES = ["latitude", "longitude", "dock_count_raw", "operation_type_id"]
STATIC_SCALED_FEATURES = ["latitude", "longitude", "dock_count_raw"]
STATIC_UNSCALED_FEATURES = ["operation_type_id"]
CATEGORICAL_FEATURES = ["station_index", "district_id"]

DYNAMIC_TRANSFORMS = {
    "rental_count": "log1p",
    "return_count": "log1p",
    "net_demand": "signed_log1p",
    "avg_duration_min": "none",
    "avg_distance_m": "none",
    "temperature": "none",
    "wind_speed": "none",
    "rainfall": "none",
    "humidity": "none",
}

TARGET_TRANSFORMS = {
    "rental_count": "log1p",
    "return_count": "log1p",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LSTM baseline arrays from preprocessed panel.")
    parser.add_argument("--preprocessed_dir", type=Path, default=Path("data/preprocessed/2025"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/lstm_baseline"))
    parser.add_argument("--sequence_length", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--train_end", type=str, default="2025-09-30 23:30:00")
    parser.add_argument("--val_end", type=str, default="2025-10-31 23:30:00")
    return parser.parse_args()


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_inputs(preprocessed_dir: Path) -> tuple[Path, pd.DataFrame, np.ndarray, dict]:
    panel_path = preprocessed_dir / "station_time_panel.parquet"
    station_metadata_path = preprocessed_dir / "station_metadata_clean.parquet"
    station_numbers_path = preprocessed_dir / "station_numbers.npy"
    summary_path = preprocessed_dir / "preprocessing_summary.json"

    required = [panel_path, station_metadata_path, station_numbers_path, summary_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required preprocessed input files: {missing}")

    station_metadata = pd.read_parquet(station_metadata_path)
    station_numbers = np.load(station_numbers_path).astype(np.int64)
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    return panel_path, station_metadata, station_numbers, summary


def build_timestamps(summary: dict) -> np.ndarray:
    timestamps = pd.date_range(
        summary["start_timestamp"],
        summary["end_timestamp"],
        freq="30min",
    ).to_numpy(dtype="datetime64[ns]")
    expected = int(summary["num_timestamps"])
    if len(timestamps) != expected:
        raise ValueError(f"Expected {expected} timestamps from summary, got {len(timestamps)}")
    return timestamps


def add_cyclic_time_features(df: pd.DataFrame) -> pd.DataFrame:
    hour_float = df["hour"].astype("float32") + df["minute"].astype("float32") / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour_float / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour_float / 24.0)
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"].astype("float32") / 7.0)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"].astype("float32") / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"].astype("float32") - 1.0) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"].astype("float32") - 1.0) / 12.0)
    return df


def build_station_mappings(
    station_metadata: pd.DataFrame,
    station_numbers: np.ndarray,
) -> tuple[dict[str, int], dict[str, int], np.ndarray, dict[str, int], np.ndarray, pd.DataFrame]:
    metadata = station_metadata.copy()
    metadata["station_number"] = metadata["station_number"].astype(np.int64)
    metadata = metadata.set_index("station_number").loc[station_numbers].reset_index()

    station_index_map = {str(int(number)): int(idx) for idx, number in enumerate(station_numbers)}

    district_values = sorted(metadata["district"].astype(str).unique().tolist())
    district_vocab = {value: idx for idx, value in enumerate(district_values)}
    district_ids = metadata["district"].astype(str).map(district_vocab).to_numpy(dtype=np.int64)

    operation_type_canonical = metadata["operation_type"].astype(str).map(canonical_operation_type)
    operation_type_vocab = {"LCD": 0, "QR": 1}
    operation_type_ids = (
        operation_type_canonical.map(operation_type_vocab).to_numpy(dtype=np.int64)
    )

    return (
        station_index_map,
        district_vocab,
        district_ids,
        operation_type_vocab,
        operation_type_ids,
        metadata,
    )


def canonical_operation_type(value: str) -> str:
    """Collapse mixed station operation labels into the binary baseline encoding.

    The LSTM baseline specification uses operation_type_id in {0, 1}. A small
    number of raw station rows are labeled "LCD,QR"; for this baseline, any
    station with QR support is assigned to the QR class.
    """
    normalized = value.replace(" ", "").upper()
    if "QR" in normalized:
        return "QR"
    if "LCD" in normalized:
        return "LCD"
    raise ValueError(f"Unknown operation_type value: {value}")


def create_chronological_splits(
    timestamps: np.ndarray,
    sequence_length: int,
    horizon: int,
    train_end: str,
    val_end: str,
) -> dict[str, list[int]]:
    ts = pd.DatetimeIndex(timestamps)
    first_valid_idx = sequence_length + horizon - 1
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)

    valid_mask = np.arange(len(ts)) >= first_valid_idx
    train = np.where(valid_mask & (ts <= train_end_ts))[0]
    val = np.where(valid_mask & (ts > train_end_ts) & (ts <= val_end_ts))[0]
    test = np.where(valid_mask & (ts > val_end_ts))[0]

    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise ValueError(
            f"Empty split detected: train={len(train)}, val={len(val)}, test={len(test)}"
        )

    return {
        "train_target_indices": train.astype(int).tolist(),
        "val_target_indices": val.astype(int).tolist(),
        "test_target_indices": test.astype(int).tolist(),
    }


def build_dynamic_tensor(
    panel_path: Path,
    output_dir: Path,
    timestamps: np.ndarray,
    station_numbers: np.ndarray,
) -> np.memmap:
    T = len(timestamps)
    S = len(station_numbers)
    dynamic_path = output_dir / "dynamic_features.npy"
    dynamic = np.lib.format.open_memmap(
        dynamic_path,
        mode="w+",
        dtype="float32",
        shape=(T, S, len(SEQUENCE_FEATURES)),
    )

    timestamp_index = pd.Index(timestamps)
    station_index = pd.Series(np.arange(S, dtype=np.int64), index=station_numbers)
    columns = [
        "timestamp",
        "station_number",
        "rental_count",
        "return_count",
        "net_demand",
        "avg_duration_min",
        "avg_distance_m",
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

    parquet_file = pq.ParquetFile(panel_path)
    for row_group in range(parquet_file.num_row_groups):
        logging.info("Building dynamic tensor from row group %s/%s", row_group + 1, parquet_file.num_row_groups)
        df = parquet_file.read_row_group(row_group, columns=columns).to_pandas()
        df = add_cyclic_time_features(df)
        time_idx = timestamp_index.get_indexer(df["timestamp"].to_numpy(dtype="datetime64[ns]"))
        station_idx = station_index.loc[df["station_number"].to_numpy(dtype=np.int64)].to_numpy()
        if (time_idx < 0).any():
            raise ValueError("Panel contains timestamps outside the expected 2025 30-minute grid.")

        values = df[SEQUENCE_FEATURES].to_numpy(dtype="float32", copy=False)
        dynamic[time_idx, station_idx, :] = values

    dynamic.flush()
    return dynamic


def build_target_tensor(
    dynamic: np.ndarray,
    output_dir: Path,
) -> tuple[np.memmap, np.memmap]:
    targets_raw = np.lib.format.open_memmap(
        output_dir / "targets_raw.npy",
        mode="w+",
        dtype="float32",
        shape=(dynamic.shape[0], dynamic.shape[1], len(TARGET_COLUMNS)),
    )
    targets = np.lib.format.open_memmap(
        output_dir / "targets.npy",
        mode="w+",
        dtype="float32",
        shape=(dynamic.shape[0], dynamic.shape[1], len(TARGET_COLUMNS)),
    )
    targets_raw[:, :, 0] = dynamic[:, :, SEQUENCE_FEATURES.index("rental_count")]
    targets_raw[:, :, 1] = dynamic[:, :, SEQUENCE_FEATURES.index("return_count")]
    targets_raw.flush()
    return targets, targets_raw


def safe_mean_std(values: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(values, dtype=np.float64))
    std = float(np.std(values, dtype=np.float64))
    if std == 0.0:
        std = 1.0
    return mean, std


def fit_dynamic_scalers_train_only(
    dynamic: np.ndarray,
    splits: dict[str, list[int]],
    horizon: int,
) -> dict[str, dict[str, float | str]]:
    max_train_target_idx = max(splits["train_target_indices"])
    last_train_input_idx = max_train_target_idx - horizon
    train_input_slice = slice(0, last_train_input_idx + 1)

    scalers: dict[str, dict[str, float | str]] = {}
    for feature in DYNAMIC_SCALED_FEATURES:
        feature_idx = SEQUENCE_FEATURES.index(feature)
        transform_name = DYNAMIC_TRANSFORMS[feature]
        transformed = apply_transform(dynamic[train_input_slice, :, feature_idx], transform_name)
        mean, std = safe_mean_std(transformed)
        scalers[feature] = {"transform": transform_name, "mean": mean, "std": std}
    return scalers


def apply_dynamic_scaling(
    dynamic: np.memmap,
    scalers: dict[str, dict[str, float | str]],
    chunk_size: int = 512,
) -> None:
    for feature, scaler in scalers.items():
        feature_idx = SEQUENCE_FEATURES.index(feature)
        transform_name = str(scaler["transform"])
        mean = float(scaler["mean"])
        std = float(scaler["std"])
        for start in range(0, dynamic.shape[0], chunk_size):
            end = min(start + chunk_size, dynamic.shape[0])
            values = dynamic[start:end, :, feature_idx]
            transformed = apply_transform(values, transform_name)
            dynamic[start:end, :, feature_idx] = standardize(transformed, mean, std).astype("float32")
    dynamic.flush()


def build_static_features(
    station_metadata: pd.DataFrame,
    operation_type_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, float | str]]]:
    raw = station_metadata[["latitude", "longitude", "dock_count_raw"]].to_numpy(dtype="float32")
    static = np.empty((len(station_metadata), len(STATIC_NUMERIC_FEATURES)), dtype="float32")
    scalers: dict[str, dict[str, float | str]] = {}

    for idx, feature in enumerate(STATIC_SCALED_FEATURES):
        mean, std = safe_mean_std(raw[:, idx])
        static[:, idx] = standardize(raw[:, idx], mean, std).astype("float32")
        scalers[feature] = {"transform": "none", "mean": mean, "std": std}

    static[:, 3] = operation_type_ids.astype("float32")
    return static, raw, scalers


def fit_target_scalers_train_only(
    targets_raw: np.ndarray,
    splits: dict[str, list[int]],
) -> dict[str, dict[str, float | str]]:
    train_target_indices = np.asarray(splits["train_target_indices"], dtype=np.int64)
    scalers: dict[str, dict[str, float | str]] = {}
    for idx, target in enumerate(TARGET_COLUMNS):
        transform_name = TARGET_TRANSFORMS[target]
        transformed = apply_transform(targets_raw[train_target_indices, :, idx], transform_name)
        mean, std = safe_mean_std(transformed)
        scalers[target] = {"transform": transform_name, "mean": mean, "std": std}
    return scalers


def apply_target_transform_and_scaling(
    targets: np.memmap,
    targets_raw: np.ndarray,
    scalers: dict[str, dict[str, float | str]],
    chunk_size: int = 512,
) -> None:
    for idx, target in enumerate(TARGET_COLUMNS):
        scaler = scalers[target]
        transform_name = str(scaler["transform"])
        mean = float(scaler["mean"])
        std = float(scaler["std"])
        for start in range(0, targets.shape[0], chunk_size):
            end = min(start + chunk_size, targets.shape[0])
            transformed = apply_transform(targets_raw[start:end, :, idx], transform_name)
            targets[start:end, :, idx] = standardize(transformed, mean, std).astype("float32")
    targets.flush()


def build_feature_config(sequence_length: int, horizon: int) -> dict:
    return {
        "task": {
            "target": TARGET_COLUMNS,
            "horizon": horizon,
            "sequence_length": sequence_length,
        },
        "sequence_features": SEQUENCE_FEATURES,
        "dynamic_scaled_features": DYNAMIC_SCALED_FEATURES,
        "dynamic_unscaled_features": DYNAMIC_UNSCALED_FEATURES,
        "static_numeric_features": STATIC_NUMERIC_FEATURES,
        "static_scaled_features": STATIC_SCALED_FEATURES,
        "static_unscaled_features": STATIC_UNSCALED_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target_columns": TARGET_COLUMNS,
        "target_transform": TARGET_TRANSFORMS,
    }


def validate_panel_inputs(
    panel_path: Path,
    station_metadata: pd.DataFrame,
    station_numbers: np.ndarray,
    timestamps: np.ndarray,
    summary: dict,
) -> None:
    parquet_file = pq.ParquetFile(panel_path)
    expected_rows = len(timestamps) * len(station_numbers)
    if parquet_file.metadata.num_rows != expected_rows:
        raise ValueError(
            f"Panel row count mismatch: got {parquet_file.metadata.num_rows}, expected {expected_rows}"
        )
    if expected_rows != int(summary["station_time_panel_rows"]):
        raise ValueError("Panel row count does not match preprocessing_summary.json")
    if not pd.Index(timestamps).is_monotonic_increasing:
        raise ValueError("Timestamps are not sorted.")

    metadata_stations = set(station_metadata["station_number"].astype(np.int64).tolist())
    missing_metadata = set(station_numbers.tolist()) - metadata_stations
    if missing_metadata:
        raise ValueError(f"station_numbers.npy has stations missing from metadata: {sorted(missing_metadata)[:10]}")

    panel_stations = set()
    for row_group in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(row_group, columns=["station_number"])
        panel_stations.update(table.column("station_number").to_numpy().astype(np.int64).tolist())
    missing_panel = set(station_numbers.tolist()) - panel_stations
    if missing_panel:
        raise ValueError(f"station_numbers.npy has stations missing from panel: {sorted(missing_panel)[:10]}")


def validate_outputs(
    output_dir: Path,
    T: int,
    S: int,
    sequence_length: int,
    horizon: int,
    splits: dict[str, list[int]],
    feature_config: dict,
) -> None:
    arrays = {
        "dynamic_features.npy": ((T, S, len(SEQUENCE_FEATURES)), np.floating),
        "targets.npy": ((T, S, len(TARGET_COLUMNS)), np.floating),
        "targets_raw.npy": ((T, S, len(TARGET_COLUMNS)), np.floating),
        "static_numeric.npy": ((S, len(STATIC_NUMERIC_FEATURES)), np.floating),
        "district_ids.npy": ((S,), np.integer),
        "operation_type_ids.npy": ((S,), np.integer),
    }
    for filename, (shape, dtype_type) in arrays.items():
        arr = np.load(output_dir / filename, mmap_mode="r")
        if arr.shape != shape:
            raise ValueError(f"{filename} shape mismatch: got {arr.shape}, expected {shape}")
        if not np.issubdtype(arr.dtype, dtype_type):
            raise ValueError(f"{filename} dtype mismatch: got {arr.dtype}")
        if np.issubdtype(arr.dtype, np.floating) and not np.isfinite(arr).all():
            raise ValueError(f"{filename} contains NaN or inf values.")

    first_valid_idx = sequence_length + horizon - 1
    ranges = []
    for split_name, key in [
        ("train", "train_target_indices"),
        ("val", "val_target_indices"),
        ("test", "test_target_indices"),
    ]:
        indices = np.asarray(splits[key], dtype=np.int64)
        if (indices < first_valid_idx).any():
            raise ValueError(f"{split_name} has target_idx earlier than first valid index.")
        input_starts = indices - horizon - sequence_length + 1
        if (input_starts < 0).any():
            raise ValueError(f"{split_name} has negative input_start_idx.")
        ranges.append((int(indices.min()), int(indices.max())))

    if not (ranges[0][1] < ranges[1][0] and ranges[1][1] < ranges[2][0]):
        raise ValueError("Train, validation, and test target index ranges overlap.")
    if feature_config["sequence_features"] != SEQUENCE_FEATURES:
        raise ValueError("Dynamic feature order does not match feature_config.json.")


def split_range(timestamps: np.ndarray, indices: list[int]) -> dict[str, str]:
    values = timestamps[np.asarray(indices, dtype=np.int64)]
    return {
        "start": str(pd.Timestamp(values[0])),
        "end": str(pd.Timestamp(values[-1])),
    }


def save_dataset_summary(
    output_dir: Path,
    T: int,
    S: int,
    sequence_length: int,
    horizon: int,
    timestamps: np.ndarray,
    splits: dict[str, list[int]],
) -> None:
    paths = {
        name: str(output_dir / name)
        for name in [
            "dynamic_features.npy",
            "targets.npy",
            "targets_raw.npy",
            "static_numeric.npy",
            "station_numbers.npy",
            "timestamps.npy",
            "district_ids.npy",
            "operation_type_ids.npy",
            "station_index_map.json",
            "district_vocab.json",
            "operation_type_vocab.json",
            "splits.json",
            "scalers.json",
            "feature_config.json",
            "dataset_summary.json",
        ]
    }
    summary = {
        "T": T,
        "S": S,
        "sequence_length": sequence_length,
        "horizon": horizon,
        "sequence_features": SEQUENCE_FEATURES,
        "target_columns": TARGET_COLUMNS,
        "static_numeric_columns": STATIC_NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "num_train_target_timestamps": len(splits["train_target_indices"]),
        "num_validation_target_timestamps": len(splits["val_target_indices"]),
        "num_test_target_timestamps": len(splits["test_target_indices"]),
        "num_train_samples": len(splits["train_target_indices"]) * S,
        "num_validation_samples": len(splits["val_target_indices"]) * S,
        "num_test_samples": len(splits["test_target_indices"]) * S,
        "train_target_date_range": split_range(timestamps, splits["train_target_indices"]),
        "validation_target_date_range": split_range(timestamps, splits["val_target_indices"]),
        "test_target_date_range": split_range(timestamps, splits["test_target_indices"]),
        "dynamic_feature_shape": [T, S, len(SEQUENCE_FEATURES)],
        "target_shape": [T, S, len(TARGET_COLUMNS)],
        "raw_target_shape": [T, S, len(TARGET_COLUMNS)],
        "static_numeric_shape": [S, len(STATIC_NUMERIC_FEATURES)],
        "district_id_shape": [S],
        "operation_type_id_shape": [S],
        "dynamic_scaled_features": DYNAMIC_SCALED_FEATURES,
        "dynamic_unscaled_features": DYNAMIC_UNSCALED_FEATURES,
        "static_scaled_features": STATIC_SCALED_FEATURES,
        "static_unscaled_features": STATIC_UNSCALED_FEATURES,
        "target_transform_info": TARGET_TRANSFORMS,
        "scaler_file_path": str(output_dir / "scalers.json"),
        "feature_config_file_path": str(output_dir / "feature_config.json"),
        "output_file_paths": paths,
    }
    write_json(output_dir / "dataset_summary.json", summary)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    panel_path, station_metadata, station_numbers, preprocessing_summary = load_inputs(args.preprocessed_dir)
    timestamps = build_timestamps(preprocessing_summary)
    T = len(timestamps)
    S = len(station_numbers)

    validate_panel_inputs(panel_path, station_metadata, station_numbers, timestamps, preprocessing_summary)
    splits = create_chronological_splits(
        timestamps,
        args.sequence_length,
        args.horizon,
        args.train_end,
        args.val_end,
    )

    (
        station_index_map,
        district_vocab,
        district_ids,
        operation_type_vocab,
        operation_type_ids,
        station_metadata_ordered,
    ) = build_station_mappings(station_metadata, station_numbers)

    logging.info("Building dynamic feature tensor with shape (%s, %s, %s)", T, S, len(SEQUENCE_FEATURES))
    dynamic = build_dynamic_tensor(panel_path, args.output_dir, timestamps, station_numbers)
    targets, targets_raw = build_target_tensor(dynamic, args.output_dir)

    # Leakage prevention: dynamic scalers use only timestamps that can appear in
    # training input windows. Target scalers use only training target timestamps.
    dynamic_scalers = fit_dynamic_scalers_train_only(dynamic, splits, args.horizon)
    apply_dynamic_scaling(dynamic, dynamic_scalers)

    target_scalers = fit_target_scalers_train_only(targets_raw, splits)
    apply_target_transform_and_scaling(targets, targets_raw, target_scalers)

    static_numeric, _static_raw, static_scalers = build_static_features(
        station_metadata_ordered,
        operation_type_ids,
    )

    np.save(args.output_dir / "static_numeric.npy", static_numeric.astype("float32"))
    np.save(args.output_dir / "station_numbers.npy", station_numbers.astype(np.int64))
    np.save(args.output_dir / "timestamps.npy", timestamps)
    np.save(args.output_dir / "district_ids.npy", district_ids.astype(np.int64))
    np.save(args.output_dir / "operation_type_ids.npy", operation_type_ids.astype(np.int64))

    scalers = {
        "dynamic": dynamic_scalers,
        "static_numeric": static_scalers,
        "target": target_scalers,
    }
    feature_config = build_feature_config(args.sequence_length, args.horizon)

    write_json(args.output_dir / "station_index_map.json", station_index_map)
    write_json(args.output_dir / "district_vocab.json", district_vocab)
    write_json(args.output_dir / "operation_type_vocab.json", operation_type_vocab)
    write_json(args.output_dir / "splits.json", splits)
    write_json(args.output_dir / "scalers.json", scalers)
    write_json(args.output_dir / "feature_config.json", feature_config)

    validate_outputs(args.output_dir, T, S, args.sequence_length, args.horizon, splits, feature_config)
    save_dataset_summary(args.output_dir, T, S, args.sequence_length, args.horizon, timestamps, splits)

    logging.info("LSTM baseline dataset complete. Outputs written to %s", args.output_dir)


if __name__ == "__main__":
    main()
