from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - checked at runtime
    raise ImportError("pyarrow is required. Install it with: pip install pyarrow") from exc

try:
    import yaml
except ImportError as exc:  # pragma: no cover - checked at runtime
    raise ImportError("PyYAML is required. Install it with: pip install pyyaml") from exc


CSV_ENCODINGS = ("utf-8", "utf-8-sig", "cp949")
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".parquet"}
EPSILON = 1.0e-6

RAW_TRIP_REQUIRED_COLUMNS = {
    "대여일시": "rental_datetime",
    "대여 대여소번호": "rental_station_number",
    "반납일시": "return_datetime",
    "반납대여소번호": "return_station_number",
    "이용시간(분)": "duration_min",
    "이용거리(M)": "distance_m",
}
RAW_TRIP_OPTIONAL_COLUMNS = {"자전거구분": "bike_type"}

STATION_REQUIRED_COLUMNS = [
    "station_number",
    "station_name",
    "district",
    "latitude",
    "longitude",
    "dock_count_raw",
    "operation_type",
]
PANEL_REQUIRED_COLUMNS = ["timestamp", "station_number", "rental_count", "return_count"]

OUTPUT_FILENAMES = [
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

EDGE_FEATURE_COLUMNS = [
    "geo_distance_km",
    "inverse_geo_distance",
    "same_district",
    "operation_pair_lcd_lcd",
    "operation_pair_lcd_qr",
    "operation_pair_qr_lcd",
    "operation_pair_qr_qr",
    "has_od_flow_fwd",
    "log_od_flow_fwd",
    "od_probability_fwd",
    "reverse_od_probability_fwd",
    "mean_duration_min_fwd",
    "mean_trip_distance_km_fwd",
    "mean_duration_lag_bins_fwd",
    "has_od_flow_rev",
    "log_od_flow_rev",
    "od_probability_rev",
    "reverse_od_probability_rev",
    "mean_duration_min_rev",
    "mean_trip_distance_km_rev",
    "mean_duration_lag_bins_rev",
    "rental_corr",
    "return_corr",
    "rental_to_return_corr",
    "best_rental_to_return_lag_norm",
    "return_to_rental_corr",
    "best_return_to_rental_lag_norm",
]

BINARY_COLUMNS = [
    "same_district",
    "operation_pair_lcd_lcd",
    "operation_pair_lcd_qr",
    "operation_pair_qr_lcd",
    "operation_pair_qr_qr",
    "has_od_flow_fwd",
    "has_od_flow_rev",
]
CONTINUOUS_COLUMNS = [name for name in EDGE_FEATURE_COLUMNS if name not in BINARY_COLUMNS]
CONTINUOUS_INDICES = [EDGE_FEATURE_COLUMNS.index(name) for name in CONTINUOUS_COLUMNS]


@dataclass(frozen=True)
class GraphBuilderConfig:
    dataset_name: str
    output_dir: Path
    raw_rental_dir: Path
    station_metadata_path: Path
    station_time_panel_path: Path
    train_start: str
    train_end: str
    k_neighbors: int
    lags: list[int]
    chunksize: int
    overwrite: bool

    @property
    def train_bounds(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        start_ts = pd.Timestamp(self.train_start).normalize()
        end_exclusive = pd.Timestamp(self.train_end).normalize() + pd.Timedelta(days=1)
        if end_exclusive <= start_ts:
            raise ValueError(f"train_end must be on or after train_start: {self.train_start} > {self.train_end}")
        return start_ts, end_exclusive

    @classmethod
    def from_mapping(cls, config: dict[str, Any]) -> "GraphBuilderConfig":
        lags = [int(value) for value in config.get("lags", [1, 2, 3, 4])]
        if not lags or any(lag <= 0 for lag in lags):
            raise ValueError("lags must be a non-empty list of positive integers.")
        k_neighbors = int(config.get("k_neighbors", 32))
        if k_neighbors <= 0:
            raise ValueError(f"k_neighbors must be positive, got {k_neighbors}")
        chunksize = int(config.get("chunksize", 500000))
        if chunksize <= 0:
            raise ValueError(f"chunksize must be positive, got {chunksize}")
        return cls(
            dataset_name=str(config.get("dataset_name", "tct_gat1_ar")),
            output_dir=Path(config.get("output_dir", "data/tct_gat_processed/tct_gat1_ar/graph")),
            raw_rental_dir=Path(config.get("raw_rental_dir", "data/raw/rentals")),
            station_metadata_path=Path(
                config.get("station_metadata_path", "data/preprocessed/station/station_metadata_clean.parquet")
            ),
            station_time_panel_path=Path(
                config.get("station_time_panel_path", "data/preprocessed/2025/station_time_panel.parquet")
            ),
            train_start=str(config.get("train_start", "2025-01-01")),
            train_end=str(config.get("train_end", "2025-09-30")),
            k_neighbors=k_neighbors,
            lags=lags,
            chunksize=chunksize,
            overwrite=bool(config.get("overwrite", False)),
        )


@dataclass
class ODFeatureMatrices:
    od_flow: np.ndarray
    log_od_flow: np.ndarray
    od_probability: np.ndarray
    reverse_od_probability: np.ndarray
    mean_duration_min: np.ndarray
    mean_trip_distance_km: np.ndarray
    mean_duration_lag_bins: np.ndarray


@dataclass
class ODStats:
    features: ODFeatureMatrices
    raw_files_used: list[str]
    raw_rows_loaded: int
    train_rows_used: int
    dropped_unknown_station: int


@dataclass
class CorrelationMatrices:
    rental_corr: np.ndarray
    return_corr: np.ndarray
    rental_to_return_corr: np.ndarray
    best_rental_to_return_lag: np.ndarray
    return_to_rental_corr: np.ndarray
    best_return_to_rental_lag: np.ndarray


@dataclass
class RelationArtifacts:
    neighbor_index: np.ndarray
    edge_attr: np.ndarray
    selected_scores: np.ndarray
    unknown_operation_edges: int


def normalize_header(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).replace("\ufeff", "")
    text = text.replace('"', "").replace("'", "")
    text = re.sub(r"\s+", "", text)
    return text.strip().lower()


def clean_station_number(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    text = text.str.replace(r"\.0$", "", regex=True)
    text = text.str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(text, errors="coerce").astype("Int64")


def read_csv_header(path: Path) -> tuple[list[str], str]:
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            header = pd.read_csv(path, nrows=0, encoding=encoding).columns.tolist()
            return header, encoding
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError(f"Could not decode {path} with encodings {CSV_ENCODINGS}: {last_error}")


def read_table(path: Path, **kwargs: Any) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        last_error: Exception | None = None
        for encoding in CSV_ENCODINGS:
            try:
                return pd.read_csv(path, encoding=encoding, **kwargs)
            except UnicodeDecodeError as exc:
                last_error = exc
        raise ValueError(f"Could not decode {path} with encodings {CSV_ENCODINGS}: {last_error}")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, **kwargs)
    if suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    raise ValueError(f"Unsupported file extension for {path}")


def find_columns(
    available_columns: Iterable[object],
    required_columns: dict[str, str],
    path: Path,
    *,
    optional_columns: dict[str, str] | None = None,
) -> dict[str, str]:
    normalized_to_original: dict[str, str] = {}
    for column in available_columns:
        normalized = normalize_header(column)
        if normalized:
            normalized_to_original[normalized] = str(column)

    selected: dict[str, str] = {}
    missing: list[str] = []
    for source_name, output_name in required_columns.items():
        original = normalized_to_original.get(normalize_header(source_name))
        if original is None:
            missing.append(source_name)
        else:
            selected[original] = output_name

    if missing:
        available = [str(column) for column in available_columns]
        raise ValueError(f"Missing required columns in {path}: {missing}. Available columns: {available}")

    for source_name, output_name in (optional_columns or {}).items():
        original = normalized_to_original.get(normalize_header(source_name))
        if original is not None:
            selected[original] = output_name
    return selected


def list_input_files(directory: Path) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Input directory does not exist: {directory}")
    files = sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(f"No supported input files found in {directory}")
    return files


def month_periods_in_range(start_ts: pd.Timestamp, end_exclusive: pd.Timestamp) -> set[pd.Period]:
    month_start = start_ts.to_period("M")
    month_end = (end_exclusive - pd.Timedelta(nanoseconds=1)).to_period("M")
    return set(pd.period_range(month_start, month_end, freq="M"))


def extract_yymm_periods(path: Path) -> list[pd.Period]:
    periods: list[pd.Period] = []
    for token in re.findall(r"(?<!\d)(\d{4})(?!\d)", path.stem):
        year = 2000 + int(token[:2])
        month = int(token[2:])
        if 1 <= month <= 12:
            periods.append(pd.Period(year=year, month=month, freq="M"))
    return periods


def filter_files_by_date_range(
    files: list[Path],
    start_ts: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> list[Path]:
    requested_months = month_periods_in_range(start_ts, end_exclusive)
    selected: list[Path] = []
    for path in files:
        periods = extract_yymm_periods(path)
        if not periods:
            selected.append(path)
            continue
        if len(periods) == 1 and periods[0] in requested_months:
            selected.append(path)
            continue
        if len(periods) >= 2:
            file_months = set(pd.period_range(min(periods), max(periods), freq="M"))
            if file_months & requested_months:
                selected.append(path)
    return selected


def haversine_distance_matrix(latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    lat_rad = np.radians(latitudes.astype(np.float64))
    lon_rad = np.radians(longitudes.astype(np.float64))
    dlat = lat_rad[:, None] - lat_rad[None, :]
    dlon = lon_rad[:, None] - lon_rad[None, :]
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_rad[:, None]) * np.cos(lat_rad[None, :]) * np.sin(dlon / 2.0) ** 2
    distance = 2.0 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return distance.astype(np.float32)


def normalize_operation_label(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).upper()
    if "LCD" in text:
        return "LCD"
    if "QR" in text:
        return "QR"
    return ""


def operation_pair_one_hot(source_ops: np.ndarray, target_ops: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source_ops)
    target = np.asarray(target_ops)
    if source.shape != target.shape:
        raise ValueError(f"source_ops and target_ops must have the same shape: {source.shape} != {target.shape}")
    out = np.zeros((source.size, 4), dtype=np.float32)
    flat_source = source.reshape(-1)
    flat_target = target.reshape(-1)
    lcd_src = flat_source == "LCD"
    qr_src = flat_source == "QR"
    lcd_tgt = flat_target == "LCD"
    qr_tgt = flat_target == "QR"
    out[:, 0] = lcd_src & lcd_tgt
    out[:, 1] = lcd_src & qr_tgt
    out[:, 2] = qr_src & lcd_tgt
    out[:, 3] = qr_src & qr_tgt
    unknown_mask = ~((lcd_src | qr_src) & (lcd_tgt | qr_tgt))
    return out, unknown_mask


def correlation_matrix(x: np.ndarray, y: np.ndarray | None = None, epsilon: float = EPSILON) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float32)
    y_arr = x_arr if y is None else np.asarray(y, dtype=np.float32)
    if x_arr.ndim != 2 or y_arr.ndim != 2:
        raise ValueError("correlation_matrix expects 2-D arrays.")
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(f"Inputs must have the same number of rows: {x_arr.shape[0]} != {y_arr.shape[0]}")
    n_rows = x_arr.shape[0]
    if n_rows == 0:
        raise ValueError("Cannot compute correlations with zero rows.")

    x_centered = x_arr - x_arr.mean(axis=0, keepdims=True)
    y_centered = y_arr - y_arr.mean(axis=0, keepdims=True)
    x_std = x_centered.std(axis=0, keepdims=True)
    y_std = y_centered.std(axis=0, keepdims=True)
    x_z = np.divide(x_centered, np.where(x_std < epsilon, 1.0, x_std), out=np.zeros_like(x_centered), where=True)
    y_z = np.divide(y_centered, np.where(y_std < epsilon, 1.0, y_std), out=np.zeros_like(y_centered), where=True)
    corr = (x_z.T @ y_z) / float(n_rows)
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def lagged_cross_correlation(
    source: np.ndarray,
    target: np.ndarray,
    lags: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    if not lags:
        raise ValueError("lags must be non-empty.")
    source_arr = np.asarray(source, dtype=np.float32)
    target_arr = np.asarray(target, dtype=np.float32)
    if source_arr.shape != target_arr.shape:
        raise ValueError(f"source and target must have identical T x S shape: {source_arr.shape} != {target_arr.shape}")

    best_corr: np.ndarray | None = None
    best_lag: np.ndarray | None = None
    for lag in lags:
        if lag <= 0 or lag >= source_arr.shape[0]:
            raise ValueError(f"Each lag must satisfy 0 < lag < T. Got lag={lag}, T={source_arr.shape[0]}")
        corr = correlation_matrix(source_arr[:-lag], target_arr[lag:])
        if best_corr is None:
            best_corr = corr
            best_lag = np.full(corr.shape, lag, dtype=np.float32)
            continue
        update_mask = corr > best_corr
        best_corr = np.where(update_mask, corr, best_corr)
        best_lag = np.where(update_mask, float(lag), best_lag)

    assert best_corr is not None and best_lag is not None
    return np.nan_to_num(best_corr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), best_lag.astype(np.float32)


def top_k_neighbors(
    score_matrix: np.ndarray,
    k_neighbors: int,
    fallback_score_matrix: np.ndarray,
    *,
    force_self_first: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(score_matrix, dtype=np.float32)
    fallback_scores = np.asarray(fallback_score_matrix, dtype=np.float32)
    if scores.shape != fallback_scores.shape or scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("score_matrix and fallback_score_matrix must be square matrices with the same shape.")
    num_stations = scores.shape[0]
    if k_neighbors > num_stations:
        raise ValueError(f"k_neighbors={k_neighbors} cannot exceed num_stations={num_stations}")

    source_ids = np.arange(num_stations, dtype=np.int64)
    neighbor_index = np.empty((num_stations, k_neighbors), dtype=np.int64)
    selected_scores = np.empty((num_stations, k_neighbors), dtype=np.float32)

    for target_idx in range(num_stations):
        selected: list[int] = []
        if force_self_first:
            selected.append(target_idx)

        col = scores[:, target_idx]
        primary_mask = np.isfinite(col) & (col > 0)
        if force_self_first:
            primary_mask[target_idx] = False
        primary_order = np.lexsort((source_ids, -np.where(primary_mask, col, -np.inf)))
        for source_idx in primary_order:
            if not primary_mask[source_idx]:
                break
            selected.append(int(source_idx))
            if len(selected) == k_neighbors:
                break

        if len(selected) < k_neighbors:
            fallback_col = fallback_scores[:, target_idx]
            fallback_mask = np.isfinite(fallback_col)
            if force_self_first:
                fallback_mask[target_idx] = False
            fallback_order = np.lexsort((source_ids, -np.where(fallback_mask, fallback_col, -np.inf)))
            selected_set = set(selected)
            for source_idx in fallback_order:
                if not fallback_mask[source_idx] or int(source_idx) in selected_set:
                    continue
                selected.append(int(source_idx))
                selected_set.add(int(source_idx))
                if len(selected) == k_neighbors:
                    break

        if len(selected) != k_neighbors:
            raise ValueError(f"Could not select {k_neighbors} neighbors for target station index {target_idx}")

        neighbor_index[target_idx] = np.asarray(selected, dtype=np.int64)
        selected_scores[target_idx] = col[neighbor_index[target_idx]]

    return neighbor_index, np.nan_to_num(selected_scores, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def build_edge_attributes(
    source_indices: np.ndarray,
    target_indices: np.ndarray,
    geo_distance_km: np.ndarray,
    inverse_geo_distance: np.ndarray,
    same_district: np.ndarray,
    operation_types: np.ndarray,
    od: ODFeatureMatrices,
    correlations: CorrelationMatrices,
    *,
    max_lag: int,
) -> tuple[np.ndarray, int]:
    sources = np.asarray(source_indices, dtype=np.int64)
    targets = np.asarray(target_indices, dtype=np.int64)
    if sources.shape != targets.shape:
        raise ValueError(f"source_indices and target_indices must have the same shape: {sources.shape} != {targets.shape}")
    shape = sources.shape
    src = sources.reshape(-1)
    tgt = targets.reshape(-1)
    features = np.zeros((src.size, len(EDGE_FEATURE_COLUMNS)), dtype=np.float32)

    op_one_hot, unknown_op_mask = operation_pair_one_hot(operation_types[src], operation_types[tgt])

    def put(name: str, values: np.ndarray) -> None:
        features[:, EDGE_FEATURE_COLUMNS.index(name)] = np.asarray(values, dtype=np.float32)

    put("geo_distance_km", geo_distance_km[src, tgt])
    put("inverse_geo_distance", inverse_geo_distance[src, tgt])
    put("same_district", same_district[src, tgt])
    features[:, 3:7] = op_one_hot

    flow_fwd = od.od_flow[src, tgt]
    put("has_od_flow_fwd", flow_fwd > 0)
    put("log_od_flow_fwd", od.log_od_flow[src, tgt])
    put("od_probability_fwd", od.od_probability[src, tgt])
    put("reverse_od_probability_fwd", od.reverse_od_probability[src, tgt])
    put("mean_duration_min_fwd", od.mean_duration_min[src, tgt])
    put("mean_trip_distance_km_fwd", od.mean_trip_distance_km[src, tgt])
    put("mean_duration_lag_bins_fwd", od.mean_duration_lag_bins[src, tgt])

    flow_rev = od.od_flow[tgt, src]
    put("has_od_flow_rev", flow_rev > 0)
    put("log_od_flow_rev", od.log_od_flow[tgt, src])
    put("od_probability_rev", od.od_probability[tgt, src])
    put("reverse_od_probability_rev", od.reverse_od_probability[tgt, src])
    put("mean_duration_min_rev", od.mean_duration_min[tgt, src])
    put("mean_trip_distance_km_rev", od.mean_trip_distance_km[tgt, src])
    put("mean_duration_lag_bins_rev", od.mean_duration_lag_bins[tgt, src])

    put("rental_corr", correlations.rental_corr[src, tgt])
    put("return_corr", correlations.return_corr[src, tgt])
    put("rental_to_return_corr", correlations.rental_to_return_corr[src, tgt])
    put("best_rental_to_return_lag_norm", correlations.best_rental_to_return_lag[src, tgt] / float(max_lag))
    put("return_to_rental_corr", correlations.return_to_rental_corr[src, tgt])
    put("best_return_to_rental_lag_norm", correlations.best_return_to_rental_lag[src, tgt] / float(max_lag))

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features.reshape(*shape, len(EDGE_FEATURE_COLUMNS)).astype(np.float32), int(unknown_op_mask.sum())


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return config


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_output_dir(output_dir: Path, overwrite: bool) -> None:
    existing = [output_dir / filename for filename in OUTPUT_FILENAMES if (output_dir / filename).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Graph output files already exist. Use --overwrite or set overwrite: true. "
            f"Existing files: {[str(path) for path in existing]}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def load_station_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing station metadata: {path}")
    metadata = pd.read_parquet(path)
    missing = set(STATION_REQUIRED_COLUMNS) - set(metadata.columns)
    if missing:
        raise ValueError(f"Station metadata missing columns: {sorted(missing)}")
    metadata = metadata[STATION_REQUIRED_COLUMNS].copy()
    if metadata[["station_number", "latitude", "longitude"]].isna().any().any():
        raise ValueError("Station metadata contains missing station_number, latitude, or longitude values.")
    metadata["station_number"] = metadata["station_number"].astype(np.int64)
    metadata = metadata.drop_duplicates("station_number", keep="first")
    return metadata.sort_values("station_number").reset_index(drop=True)


def initialize_od_accumulators(num_stations: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flow = np.zeros((num_stations, num_stations), dtype=np.int32)
    duration_sum = np.zeros((num_stations, num_stations), dtype=np.float64)
    distance_sum_m = np.zeros((num_stations, num_stations), dtype=np.float64)
    return flow, duration_sum, distance_sum_m


def process_trip_chunk(
    df: pd.DataFrame,
    station_to_idx: dict[int, int],
    start_ts: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    od_flow: np.ndarray,
    duration_sum: np.ndarray,
    distance_sum_m: np.ndarray,
) -> tuple[int, int, int]:
    required = list(RAW_TRIP_REQUIRED_COLUMNS.values())
    if set(required).issubset(df.columns):
        trips = df[required].copy()
    else:
        selected = find_columns(df.columns, RAW_TRIP_REQUIRED_COLUMNS, Path("<dataframe>"))
        trips = df.rename(columns=selected)[required].copy()

    raw_rows = len(trips)
    trips["rental_datetime"] = pd.to_datetime(trips["rental_datetime"], errors="coerce")
    trips["return_datetime"] = pd.to_datetime(trips["return_datetime"], errors="coerce")
    trips["rental_station_number"] = clean_station_number(trips["rental_station_number"])
    trips["return_station_number"] = clean_station_number(trips["return_station_number"])
    trips["duration_min"] = pd.to_numeric(trips["duration_min"], errors="coerce")
    trips["distance_m"] = pd.to_numeric(trips["distance_m"], errors="coerce")
    trips = trips.dropna(
        subset=[
            "rental_datetime",
            "return_datetime",
            "rental_station_number",
            "return_station_number",
            "duration_min",
            "distance_m",
        ]
    )
    if trips.empty:
        return raw_rows, 0, 0

    trips["rental_station_number"] = trips["rental_station_number"].astype(np.int64)
    trips["return_station_number"] = trips["return_station_number"].astype(np.int64)
    train_mask = (
        (trips["rental_datetime"] >= start_ts)
        & (trips["rental_datetime"] < end_exclusive)
        & (trips["return_datetime"] >= start_ts)
        & (trips["return_datetime"] < end_exclusive)
    )
    trips = trips[train_mask].copy()
    if trips.empty:
        return raw_rows, 0, 0

    station_set = set(station_to_idx)
    known_mask = trips["rental_station_number"].isin(station_set) & trips["return_station_number"].isin(station_set)
    dropped_unknown = int((~known_mask).sum())
    trips = trips[known_mask]
    if trips.empty:
        return raw_rows, 0, dropped_unknown

    src = trips["rental_station_number"].map(station_to_idx).to_numpy(dtype=np.int64)
    tgt = trips["return_station_number"].map(station_to_idx).to_numpy(dtype=np.int64)
    duration = trips["duration_min"].to_numpy(dtype=np.float64)
    distance = trips["distance_m"].to_numpy(dtype=np.float64)

    np.add.at(od_flow, (src, tgt), 1)
    np.add.at(duration_sum, (src, tgt), duration)
    np.add.at(distance_sum_m, (src, tgt), distance)
    return raw_rows, int(len(trips)), dropped_unknown


def finalize_od_features(
    od_flow: np.ndarray,
    duration_sum: np.ndarray,
    distance_sum_m: np.ndarray,
    *,
    max_lag: int,
) -> ODFeatureMatrices:
    flow_float = od_flow.astype(np.float32)
    log_od_flow = np.log1p(flow_float).astype(np.float32)
    total_rentals_from = flow_float.sum(axis=1, keepdims=True)
    total_returns_to = flow_float.sum(axis=0, keepdims=True)
    od_probability = np.divide(
        flow_float,
        total_rentals_from,
        out=np.zeros_like(flow_float),
        where=total_rentals_from > 0,
    )
    reverse_od_probability = np.divide(
        flow_float,
        total_returns_to,
        out=np.zeros_like(flow_float),
        where=total_returns_to > 0,
    )
    mean_duration = np.divide(
        duration_sum,
        od_flow,
        out=np.zeros_like(duration_sum),
        where=od_flow > 0,
    ).astype(np.float32)
    mean_distance_km = np.divide(
        distance_sum_m,
        od_flow * 1000.0,
        out=np.zeros_like(distance_sum_m),
        where=od_flow > 0,
    ).astype(np.float32)
    mean_duration_lag_bins = np.clip(mean_duration / 30.0, 0.0, float(max_lag)).astype(np.float32)
    return ODFeatureMatrices(
        od_flow=od_flow,
        log_od_flow=log_od_flow,
        od_probability=od_probability.astype(np.float32),
        reverse_od_probability=reverse_od_probability.astype(np.float32),
        mean_duration_min=mean_duration,
        mean_trip_distance_km=mean_distance_km,
        mean_duration_lag_bins=mean_duration_lag_bins,
    )


def build_od_features(config: GraphBuilderConfig, station_numbers: np.ndarray) -> ODStats:
    start_ts, end_exclusive = config.train_bounds
    files = filter_files_by_date_range(list_input_files(config.raw_rental_dir), start_ts, end_exclusive)
    if not files:
        raise FileNotFoundError(
            f"No rental files overlap train range {start_ts.date()} to {(end_exclusive - pd.Timedelta(days=1)).date()}."
        )

    station_to_idx = {int(number): idx for idx, number in enumerate(station_numbers.tolist())}
    od_flow, duration_sum, distance_sum_m = initialize_od_accumulators(len(station_numbers))
    raw_rows_loaded = 0
    train_rows_used = 0
    dropped_unknown = 0

    for path in files:
        logging.info("Aggregating train-period OD trips from %s", path)
        if path.suffix.lower() == ".csv":
            columns, encoding = read_csv_header(path)
            selected = find_columns(
                columns,
                RAW_TRIP_REQUIRED_COLUMNS,
                path,
                optional_columns=RAW_TRIP_OPTIONAL_COLUMNS,
            )
            reader = pd.read_csv(
                path,
                encoding=encoding,
                usecols=list(selected.keys()),
                chunksize=config.chunksize,
                low_memory=False,
            )
            for chunk in reader:
                chunk = chunk.rename(columns=selected)
                raw_rows, used_rows, unknown_rows = process_trip_chunk(
                    chunk,
                    station_to_idx,
                    start_ts,
                    end_exclusive,
                    od_flow,
                    duration_sum,
                    distance_sum_m,
                )
                raw_rows_loaded += raw_rows
                train_rows_used += used_rows
                dropped_unknown += unknown_rows
        else:
            df = read_table(path)
            selected = find_columns(
                df.columns,
                RAW_TRIP_REQUIRED_COLUMNS,
                path,
                optional_columns=RAW_TRIP_OPTIONAL_COLUMNS,
            )
            raw_rows, used_rows, unknown_rows = process_trip_chunk(
                df.rename(columns=selected),
                station_to_idx,
                start_ts,
                end_exclusive,
                od_flow,
                duration_sum,
                distance_sum_m,
            )
            raw_rows_loaded += raw_rows
            train_rows_used += used_rows
            dropped_unknown += unknown_rows

    features = finalize_od_features(od_flow, duration_sum, distance_sum_m, max_lag=max(config.lags))
    return ODStats(
        features=features,
        raw_files_used=[str(path) for path in files],
        raw_rows_loaded=int(raw_rows_loaded),
        train_rows_used=int(train_rows_used),
        dropped_unknown_station=int(dropped_unknown),
    )


def load_train_count_matrices(
    panel_path: Path,
    station_numbers: np.ndarray,
    start_ts: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray]:
    if not panel_path.exists():
        raise FileNotFoundError(f"Missing station-time panel: {panel_path}")
    timestamps = pd.date_range(start_ts, end_exclusive - pd.Timedelta(minutes=30), freq="30min")
    time_index = pd.Series(np.arange(len(timestamps), dtype=np.int64), index=timestamps)
    station_index = pd.Series(np.arange(len(station_numbers), dtype=np.int64), index=station_numbers.astype(np.int64))
    rental_counts = np.zeros((len(timestamps), len(station_numbers)), dtype=np.float32)
    return_counts = np.zeros_like(rental_counts)

    parquet_file = pq.ParquetFile(panel_path)
    for row_group in range(parquet_file.num_row_groups):
        logging.info("Reading station-time panel row group %s/%s", row_group + 1, parquet_file.num_row_groups)
        df = parquet_file.read_row_group(row_group, columns=PANEL_REQUIRED_COLUMNS).to_pandas()
        missing = set(PANEL_REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Panel missing columns: {sorted(missing)}")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] < end_exclusive)]
        if df.empty:
            continue
        df["station_number"] = df["station_number"].astype(np.int64)
        known = df["station_number"].isin(station_index.index)
        if not known.all():
            unknown = sorted(df.loc[~known, "station_number"].unique().tolist())[:10]
            raise ValueError(f"Panel contains stations not in metadata: {unknown}")
        t_idx = time_index.loc[df["timestamp"]].to_numpy(dtype=np.int64)
        s_idx = station_index.loc[df["station_number"]].to_numpy(dtype=np.int64)
        rental_counts[t_idx, s_idx] = pd.to_numeric(df["rental_count"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        return_counts[t_idx, s_idx] = pd.to_numeric(df["return_count"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)

    return rental_counts, return_counts


def build_correlations(
    panel_path: Path,
    station_numbers: np.ndarray,
    start_ts: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    lags: list[int],
) -> CorrelationMatrices:
    rental_counts, return_counts = load_train_count_matrices(panel_path, station_numbers, start_ts, end_exclusive)
    logging.info("Computing log-count correlation matrices")
    rental_log = np.log1p(rental_counts).astype(np.float32)
    return_log = np.log1p(return_counts).astype(np.float32)
    rental_corr = correlation_matrix(rental_log)
    return_corr = correlation_matrix(return_log)
    np.fill_diagonal(rental_corr, 1.0)
    np.fill_diagonal(return_corr, 1.0)
    logging.info("Computing lagged rental-to-return correlations for lags=%s", lags)
    rental_to_return_corr, best_rental_to_return_lag = lagged_cross_correlation(rental_log, return_log, lags)
    logging.info("Computing lagged return-to-rental correlations for lags=%s", lags)
    return_to_rental_corr, best_return_to_rental_lag = lagged_cross_correlation(return_log, rental_log, lags)
    return CorrelationMatrices(
        rental_corr=rental_corr,
        return_corr=return_corr,
        rental_to_return_corr=rental_to_return_corr,
        best_rental_to_return_lag=best_rental_to_return_lag,
        return_to_rental_corr=return_to_rental_corr,
        best_return_to_rental_lag=best_return_to_rental_lag,
    )


def normalized_log_flow(log_od_flow: np.ndarray) -> np.ndarray:
    max_value = float(np.max(log_od_flow))
    if max_value <= 0:
        return np.zeros_like(log_od_flow, dtype=np.float32)
    return (log_od_flow / max_value).astype(np.float32)


def build_relation_scores(
    od: ODFeatureMatrices,
    correlations: CorrelationMatrices,
    inverse_geo_distance: np.ndarray,
    same_district: np.ndarray,
) -> dict[str, np.ndarray]:
    log_flow_norm = normalized_log_flow(od.log_od_flow)
    rd_score = (
        2.0 * log_flow_norm
        + 1.0 * od.od_probability
        + 1.0 * od.reverse_od_probability
        + 0.5 * np.maximum(correlations.rental_to_return_corr, 0.0)
        + 0.2 * inverse_geo_distance
    )

    log_flow_norm_rev = log_flow_norm.T
    dr_score = (
        1.5 * log_flow_norm_rev
        + 0.8 * od.od_probability.T
        + 0.8 * od.reverse_od_probability.T
        + 0.7 * np.maximum(correlations.return_to_rental_corr, 0.0)
        + 0.2 * inverse_geo_distance
    )

    rr_score = (
        1.0 * np.maximum(correlations.rental_corr, 0.0)
        + 0.3 * inverse_geo_distance
        + 0.2 * same_district
    )
    dd_score = (
        1.0 * np.maximum(correlations.return_corr, 0.0)
        + 0.3 * inverse_geo_distance
        + 0.2 * same_district
    )
    return {
        "rr": np.nan_to_num(rr_score, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        "dd": np.nan_to_num(dd_score, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        "rd": np.nan_to_num(rd_score, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        "dr": np.nan_to_num(dr_score, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
    }


def build_relation_artifacts(
    relation_scores: dict[str, np.ndarray],
    inverse_geo_distance: np.ndarray,
    geo_distance_km: np.ndarray,
    same_district: np.ndarray,
    operation_types: np.ndarray,
    od: ODFeatureMatrices,
    correlations: CorrelationMatrices,
    *,
    k_neighbors: int,
    max_lag: int,
) -> dict[str, RelationArtifacts]:
    artifacts: dict[str, RelationArtifacts] = {}
    for relation, force_self in [("rr", True), ("dd", True), ("rd", False), ("dr", False)]:
        logging.info("Selecting top-%s neighbors for %s relation", k_neighbors, relation)
        neighbor_index, selected_scores = top_k_neighbors(
            relation_scores[relation],
            k_neighbors,
            inverse_geo_distance,
            force_self_first=force_self,
        )
        target_indices = np.broadcast_to(
            np.arange(neighbor_index.shape[0], dtype=np.int64)[:, None],
            neighbor_index.shape,
        )
        edge_attr, unknown_op_edges = build_edge_attributes(
            neighbor_index,
            target_indices,
            geo_distance_km,
            inverse_geo_distance,
            same_district,
            operation_types,
            od,
            correlations,
            max_lag=max_lag,
        )
        artifacts[relation] = RelationArtifacts(
            neighbor_index=neighbor_index,
            edge_attr=edge_attr,
            selected_scores=selected_scores,
            unknown_operation_edges=unknown_op_edges,
        )
    return artifacts


def normalize_edge_attributes(artifacts: dict[str, RelationArtifacts]) -> dict[str, dict[str, float]]:
    stacked = np.concatenate(
        [artifact.edge_attr[:, :, CONTINUOUS_INDICES].reshape(-1, len(CONTINUOUS_INDICES)) for artifact in artifacts.values()],
        axis=0,
    )
    means = stacked.mean(axis=0)
    stds = stacked.std(axis=0)
    stds = np.where(stds < EPSILON, 1.0, stds)
    for artifact in artifacts.values():
        values = artifact.edge_attr[:, :, CONTINUOUS_INDICES]
        artifact.edge_attr[:, :, CONTINUOUS_INDICES] = (values - means[None, None, :]) / stds[None, None, :]
        artifact.edge_attr[:] = np.nan_to_num(artifact.edge_attr, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        column: {"mean": float(mean), "std": float(std)}
        for column, mean, std in zip(CONTINUOUS_COLUMNS, means.tolist(), stds.tolist())
    }


def summarize_relation(artifact: RelationArtifacts, relation: str) -> dict[str, Any]:
    scores = artifact.selected_scores
    notes = {
        "rr": "Self-edge is forced at k=0; remaining neighbors rank rental correlation, distance, and district.",
        "dd": "Self-edge is forced at k=0; remaining neighbors rank return correlation, distance, and district.",
        "rd": "Ranks rental-origin to return-destination OD evidence, lagged correlation, and distance.",
        "dr": "Ranks reverse physical OD evidence, return-to-rental lagged correlation, and distance.",
    }
    return {
        "neighbor_index_shape": list(artifact.neighbor_index.shape),
        "edge_attr_shape": list(artifact.edge_attr.shape),
        "mean_score": float(np.mean(scores)),
        "min_score": float(np.min(scores)),
        "max_score": float(np.max(scores)),
        "notes": notes[relation],
    }


def save_artifacts(
    output_dir: Path,
    config: GraphBuilderConfig,
    artifacts: dict[str, RelationArtifacts],
    normalization: dict[str, dict[str, float]],
    od_stats: ODStats,
    num_stations: int,
) -> None:
    for relation, artifact in artifacts.items():
        np.save(output_dir / f"neighbor_index_{relation}.npy", artifact.neighbor_index.astype(np.int64))
        np.save(output_dir / f"edge_attr_{relation}.npy", artifact.edge_attr.astype(np.float32))
    write_json(output_dir / "edge_feature_columns.json", EDGE_FEATURE_COLUMNS)

    summary = {
        "dataset_name": config.dataset_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "train_start": config.train_start,
        "train_end": config.train_end,
        "num_stations": int(num_stations),
        "k_neighbors": int(config.k_neighbors),
        "lags": [int(lag) for lag in config.lags],
        "edge_feature_columns": EDGE_FEATURE_COLUMNS,
        "continuous_columns": CONTINUOUS_COLUMNS,
        "binary_columns": BINARY_COLUMNS,
        "normalization": normalization,
        "input_paths": {
            "raw_rental_dir": str(config.raw_rental_dir),
            "station_metadata_path": str(config.station_metadata_path),
            "station_time_panel_path": str(config.station_time_panel_path),
        },
        "raw_rental_files_used": od_stats.raw_files_used,
        "num_raw_trip_rows_loaded": int(od_stats.raw_rows_loaded),
        "num_train_trip_rows_used": int(od_stats.train_rows_used),
        "num_trips_dropped_unknown_station": int(od_stats.dropped_unknown_station),
        "num_unknown_operation_type_edges": int(sum(artifact.unknown_operation_edges for artifact in artifacts.values())),
        "relation_summaries": {
            relation: summarize_relation(artifact, relation) for relation, artifact in artifacts.items()
        },
    }
    write_json(output_dir / "graph_summary.json", summary)


def build_graph(config: GraphBuilderConfig) -> None:
    start_ts, end_exclusive = config.train_bounds
    validate_output_dir(config.output_dir, config.overwrite)

    logging.info("Loading station metadata from %s", config.station_metadata_path)
    station_metadata = load_station_metadata(config.station_metadata_path)
    num_stations = len(station_metadata)
    if config.k_neighbors > num_stations:
        raise ValueError(f"k_neighbors={config.k_neighbors} exceeds num_stations={num_stations}")
    station_numbers = station_metadata["station_number"].to_numpy(dtype=np.int64)

    logging.info("Computing geographic and station static matrices")
    geo_distance_km = haversine_distance_matrix(
        station_metadata["latitude"].to_numpy(dtype=np.float64),
        station_metadata["longitude"].to_numpy(dtype=np.float64),
    )
    inverse_geo_distance = (1.0 / (1.0 + geo_distance_km)).astype(np.float32)
    districts = station_metadata["district"].astype("string").fillna("").to_numpy(dtype=str)
    same_district = ((districts[:, None] == districts[None, :]) & (districts[:, None] != "")).astype(np.float32)
    operation_types = np.asarray([normalize_operation_label(value) for value in station_metadata["operation_type"]], dtype=object)

    logging.info("Building train-only OD feature matrices")
    od_stats = build_od_features(config, station_numbers)

    logging.info("Building train-only station-time correlation matrices")
    correlations = build_correlations(
        config.station_time_panel_path,
        station_numbers,
        start_ts,
        end_exclusive,
        config.lags,
    )

    logging.info("Scoring graph relations")
    relation_scores = build_relation_scores(
        od_stats.features,
        correlations,
        inverse_geo_distance,
        same_district,
    )
    artifacts = build_relation_artifacts(
        relation_scores,
        inverse_geo_distance,
        geo_distance_km,
        same_district,
        operation_types,
        od_stats.features,
        correlations,
        k_neighbors=config.k_neighbors,
        max_lag=max(config.lags),
    )

    logging.info("Normalizing continuous edge features over selected sparse edges")
    normalization = normalize_edge_attributes(artifacts)

    logging.info("Writing graph artifacts to %s", config.output_dir)
    save_artifacts(config.output_dir, config, artifacts, normalization, od_stats, num_stations)
    logging.info("Done")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fixed sparse graph artifacts for TCT-GAT1-AR.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/tct_gat/tct_gat1_ar_graph.yaml"),
        help="Path to graph-builder YAML config.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Override config overwrite and replace existing graph artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    config_dict = load_config(args.config)
    if args.overwrite:
        config_dict["overwrite"] = True
    config = GraphBuilderConfig.from_mapping(config_dict)
    build_graph(config)


if __name__ == "__main__":
    main()
