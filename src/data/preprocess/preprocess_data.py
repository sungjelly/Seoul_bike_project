from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - checked at runtime
    raise ImportError(
        "pyarrow is required to write parquet outputs. Install it with: pip install pyarrow"
    ) from exc


DEFAULT_START_DATE = "2025-01-01"
DEFAULT_END_DATE = "2025-12-31"
CSV_ENCODINGS = ("utf-8", "utf-8-sig", "cp949")
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".parquet"}

RENTAL_REQUIRED_COLUMNS = {
    "대여일시": "rental_datetime",
    "대여 대여소번호": "rental_station_number",
    "반납일시": "return_datetime",
    "반납대여소번호": "return_station_number",
    "이용시간(분)": "duration_min",
    "이용거리(M)": "distance_m",
}

WEATHER_REQUIRED_COLUMNS = {
    "일시": "timestamp",
    "기온(°C)": "temperature",
    "풍속(m/s)": "wind_speed",
    "강수량(mm)": "rainfall",
    "습도(%)": "humidity",
}

STATION_OUTPUT_COLUMNS = [
    "station_number",
    "station_name",
    "district",
    "latitude",
    "longitude",
    "lcd_dock_count",
    "qr_dock_count",
    "dock_count_raw",
    "operation_type",
]

PANEL_COLUMNS = [
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
    "latitude",
    "longitude",
    "district",
    "dock_count_raw",
    "operation_type",
]

STATION_OUTPUT_FILES = [
    "station_metadata_clean.parquet",
    "station_numbers.npy",
    "station_coords.npy",
    "station_districts.json",
]

TIME_OUTPUT_FILES = [
    "weather_30min.parquet",
    "station_time_panel.parquet",
    "feature_columns.json",
    "preprocessing_summary.json",
]


def normalize_header(value: object) -> str:
    """Normalize Korean headers with line breaks, quotes, and extra whitespace."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value)
    text = text.replace("\ufeff", "")
    text = text.replace('"', "").replace("'", "")
    text = re.sub(r"\s+", "", text)
    return text.strip().lower()


def normalize_station_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return normalize_header(value)


def read_csv_header(path: Path) -> tuple[list[str], str]:
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            header = pd.read_csv(path, nrows=0, encoding=encoding).columns.tolist()
            return header, encoding
        except UnicodeDecodeError as exc:
            last_error = exc
    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"Could not decode {path} with encodings {CSV_ENCODINGS}: {last_error}",
    )


def read_table(path: Path, **kwargs) -> pd.DataFrame:
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


def list_input_files(directory: Path) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Input directory does not exist: {directory}")
    files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(f"No supported input files found in {directory}")
    return files


def parse_date_range(start_date: str, end_date: str) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.DatetimeIndex]:
    start_day = pd.Timestamp(start_date).normalize()
    end_day = pd.Timestamp(end_date).normalize()
    if end_day < start_day:
        raise ValueError(f"end-date must be on or after start-date: {start_date} > {end_date}")
    start_ts = start_day
    end_exclusive = end_day + pd.Timedelta(days=1)
    panel_end_ts = end_exclusive - pd.Timedelta(minutes=30)
    timestamp_grid = pd.date_range(start_ts, panel_end_ts, freq="30min")
    return start_ts, end_exclusive, panel_end_ts, timestamp_grid


def validate_output_dir(output_dir: Path, overwrite: bool) -> None:
    existing = [output_dir / filename for filename in TIME_OUTPUT_FILES if (output_dir / filename).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Output files already exist. Use --overwrite to replace them, or choose a new "
            f"--output-dir. Existing files: {[str(path) for path in existing]}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def station_output_paths(station_output_dir: Path) -> dict[str, Path]:
    return {filename: station_output_dir / filename for filename in STATION_OUTPUT_FILES}


def validate_station_output_dir(station_output_dir: Path, rebuild_station: bool) -> bool:
    paths = station_output_paths(station_output_dir)
    existing = [path for path in paths.values() if path.exists()]
    if rebuild_station:
        station_output_dir.mkdir(parents=True, exist_ok=True)
        return False
    if not existing:
        station_output_dir.mkdir(parents=True, exist_ok=True)
        return False
    if len(existing) != len(paths):
        missing = [str(path) for path in paths.values() if not path.exists()]
        raise FileNotFoundError(
            "Found partial shared station outputs. Use --rebuild-station to recreate them, "
            f"or restore the missing files: {missing}"
        )
    return True


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
    selected = []
    for path in files:
        periods = extract_yymm_periods(path)
        # Keep files with no date token because their rows can still be filtered
        # by parsed timestamps. This preserves correctness for unexpected names.
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


def find_columns(
    available_columns: Iterable[object],
    required_columns: dict[str, str],
    path: Path,
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
        raise ValueError(
            f"Missing required columns in {path}: {missing}. Available columns: {available}"
        )
    return selected


def clean_station_number(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    text = text.str.replace(r"\.0$", "", regex=True)
    text = text.str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(text, errors="coerce").astype("Int64")


def load_station_metadata(raw_dir: Path) -> pd.DataFrame:
    station_candidates = []
    if raw_dir.exists():
        station_candidates.extend(
            path
            for path in raw_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
            and normalize_header(path.stem).startswith("station")
        )
    station_dir = raw_dir / "station"
    if station_dir.exists():
        station_candidates.extend(
            path
            for path in station_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    if not station_candidates:
        raise FileNotFoundError(
            f"No station metadata file found under {raw_dir} or {raw_dir / 'station'}"
        )

    path = sorted(station_candidates)[0]
    logging.info("Loading station metadata: %s", path)

    # Read with the first row as headers, then infer split/merged header columns from
    # the first few rows. Seoul station files often store district/address/lat/lon
    # under a merged "소재지(위치)" header and dock counts under split LCD/QR columns.
    df = read_table(path, header=0)
    column_roles = infer_station_column_roles(df, path)

    selected = pd.DataFrame(
        {
            output_name: df[source_column]
            for output_name, source_column in column_roles.items()
        }
    )

    selected["station_number"] = clean_station_number(selected["station_number"])
    selected["latitude"] = pd.to_numeric(selected["latitude"], errors="coerce")
    selected["longitude"] = pd.to_numeric(selected["longitude"], errors="coerce")
    selected["lcd_dock_count"] = pd.to_numeric(selected["lcd_dock_count"], errors="coerce").fillna(0)
    selected["qr_dock_count"] = pd.to_numeric(selected["qr_dock_count"], errors="coerce").fillna(0)
    selected["dock_count_raw"] = selected["lcd_dock_count"] + selected["qr_dock_count"]

    for column in ("station_name", "district", "operation_type"):
        selected[column] = selected[column].astype("string").str.strip()
        selected[column] = selected[column].replace({"": pd.NA})

    selected = selected.dropna(
        subset=["station_number", "station_name", "district", "latitude", "longitude"]
    )
    selected["station_number"] = selected["station_number"].astype("int64")
    selected = selected.drop_duplicates("station_number", keep="first")
    selected = selected[STATION_OUTPUT_COLUMNS].sort_values("station_number").reset_index(drop=True)
    return selected


def load_shared_station_metadata(station_output_dir: Path) -> pd.DataFrame:
    paths = station_output_paths(station_output_dir)
    station_metadata = pd.read_parquet(paths["station_metadata_clean.parquet"])
    station_numbers = np.load(paths["station_numbers.npy"])
    station_coords = np.load(paths["station_coords.npy"])
    station_districts = json.loads(
        paths["station_districts.json"].read_text(encoding="utf-8")
    )

    expected_columns = set(STATION_OUTPUT_COLUMNS)
    missing_columns = expected_columns - set(station_metadata.columns)
    if missing_columns:
        raise ValueError(
            "Shared station metadata is missing required columns: "
            f"{sorted(missing_columns)}"
        )
    if len(station_metadata) != len(station_numbers):
        raise ValueError(
            "Shared station metadata row count does not match station_numbers.npy length: "
            f"{len(station_metadata)} != {len(station_numbers)}"
        )
    if station_coords.shape != (len(station_numbers), 2):
        raise ValueError(
            "station_coords.npy must have shape (num_stations, 2): "
            f"found {station_coords.shape}"
        )

    metadata_numbers = station_metadata["station_number"].to_numpy(dtype=np.int64)
    if not np.array_equal(metadata_numbers, station_numbers.astype(np.int64)):
        raise ValueError("station_numbers.npy does not match station_metadata_clean.parquet order.")

    metadata_coords = station_metadata[["latitude", "longitude"]].to_numpy(dtype=np.float64)
    if not np.allclose(metadata_coords, station_coords.astype(np.float64), equal_nan=True):
        raise ValueError("station_coords.npy does not match station_metadata_clean.parquet.")

    expected_districts = {str(int(number)) for number in metadata_numbers}
    if set(station_districts) != expected_districts:
        raise ValueError("station_districts.json keys do not match station numbers.")

    return station_metadata[STATION_OUTPUT_COLUMNS].copy()


def write_shared_station_metadata(station_output_dir: Path, station_metadata: pd.DataFrame) -> None:
    station_output_dir.mkdir(parents=True, exist_ok=True)
    station_metadata = station_metadata[STATION_OUTPUT_COLUMNS].sort_values("station_number").reset_index(drop=True)
    station_numbers = station_metadata["station_number"].to_numpy(dtype=np.int64)
    station_coords = station_metadata[["latitude", "longitude"]].to_numpy(dtype=np.float64)
    station_districts = {
        str(int(row.station_number)): row.district for row in station_metadata.itertuples(index=False)
    }

    station_metadata.to_parquet(station_output_dir / "station_metadata_clean.parquet", index=False)
    np.save(station_output_dir / "station_numbers.npy", station_numbers)
    np.save(station_output_dir / "station_coords.npy", station_coords)
    write_json(station_output_dir / "station_districts.json", station_districts)


def infer_station_column_roles(df: pd.DataFrame, path: Path) -> dict[str, str]:
    normalized_columns = {normalize_header(column): column for column in df.columns}

    def exact(*names: str) -> str | None:
        for name in names:
            found = normalized_columns.get(normalize_header(name))
            if found is not None:
                return str(found)
        return None

    roles: dict[str, str | None] = {
        "station_number": exact("대여소 번호", "대여소번호"),
        "station_name": exact("보관소(대여소)명", "대여소명"),
        "operation_type": exact("운영방식"),
        "district": exact("자치구"),
        "latitude": exact("위도"),
        "longitude": exact("경도"),
        "lcd_dock_count": exact("LCD 거치대수", "LCD거치대수"),
        "qr_dock_count": exact("QR 거치대수", "QR거치대수"),
    }

    probe = df.head(8)
    for column in df.columns:
        normalized_column = normalize_header(column)
        probe_values = {normalize_station_text(value) for value in probe[column].dropna().tolist()}
        combined = normalized_column + "|" + "|".join(sorted(probe_values))

        if roles["district"] is None and "자치구" in combined:
            roles["district"] = str(column)
        if roles["latitude"] is None and "위도" in combined:
            roles["latitude"] = str(column)
        if roles["longitude"] is None and "경도" in combined:
            roles["longitude"] = str(column)
        if roles["lcd_dock_count"] is None and "lcd" in combined and "거치" in combined:
            roles["lcd_dock_count"] = str(column)
        if roles["qr_dock_count"] is None and "qr" in combined and "거치" in combined:
            roles["qr_dock_count"] = str(column)

    missing = [role for role, column in roles.items() if column is None]
    if missing:
        available = [str(column) for column in df.columns]
        preview = df.head(8).to_dict(orient="list")
        raise ValueError(
            f"Could not infer station metadata columns in {path}. Missing roles: {missing}. "
            f"Available columns: {available}. First rows: {preview}"
        )
    return {role: str(column) for role, column in roles.items() if column is not None}


def load_weather(
    weather_dir: Path,
    output_grid: pd.DatetimeIndex,
    start_ts: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> tuple[pd.DataFrame, int]:
    files = filter_files_by_date_range(list_input_files(weather_dir), start_ts, end_exclusive)
    if not files:
        raise FileNotFoundError(
            f"No weather files appear to overlap requested range {start_ts.date()} to "
            f"{(end_exclusive - pd.Timedelta(days=1)).date()}."
        )
    frames = []
    rows_loaded = 0
    for path in files:
        logging.info("Loading weather file: %s", path)
        if path.suffix.lower() == ".csv":
            columns, encoding = read_csv_header(path)
            selected = find_columns(columns, WEATHER_REQUIRED_COLUMNS, path)
            df = pd.read_csv(path, encoding=encoding, usecols=list(selected.keys()))
        else:
            header_df = read_table(path, nrows=0)
            selected = find_columns(header_df.columns, WEATHER_REQUIRED_COLUMNS, path)
            df = read_table(path, usecols=list(selected.keys()))
        rows_loaded += len(df)
        df = df.rename(columns=selected)
        frames.append(df)

    weather = pd.concat(frames, ignore_index=True)
    weather["timestamp"] = pd.to_datetime(weather["timestamp"], errors="coerce")
    for column in ("temperature", "wind_speed", "rainfall", "humidity"):
        weather[column] = pd.to_numeric(weather[column], errors="coerce")
    weather = weather.dropna(subset=["timestamp"])
    weather = weather[(weather["timestamp"] >= start_ts) & (weather["timestamp"] < end_exclusive)]
    if weather.empty:
        raise ValueError(
            f"No weather rows found in requested range: {start_ts} <= timestamp < {end_exclusive}"
        )

    hourly_avg = (
        weather.groupby("timestamp", as_index=True)[
            ["temperature", "wind_speed", "rainfall", "humidity"]
        ]
        .mean()
        .sort_index()
    )
    weather_30min = hourly_avg.reindex(output_grid).ffill()
    if weather_30min.iloc[0].isna().any():
        raise ValueError(
            "Weather data does not cover the first requested timestamp. "
            f"Need weather at or before {output_grid[0]}."
        )
    weather_30min.index.name = "timestamp"
    weather_30min = weather_30min.reset_index()
    return weather_30min, rows_loaded


def process_rentals(
    rental_dir: Path,
    station_numbers: set[int],
    chunksize: int,
    start_ts: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int], set[int]]:
    files = filter_files_by_date_range(list_input_files(rental_dir), start_ts, end_exclusive)
    if not files:
        raise FileNotFoundError(
            f"No rental files appear to overlap requested range {start_ts.date()} to "
            f"{(end_exclusive - pd.Timedelta(days=1)).date()}."
        )
    rental_aggregates: list[pd.DataFrame] = []
    return_aggregates: list[pd.DataFrame] = []
    trip_aggregates: list[pd.DataFrame] = []
    used_stations: set[int] = set()
    stats = {
        "raw_rental_rows_loaded": 0,
        "rental_rows_after_cleaning": 0,
        "rental_rows_relevant_to_requested_period": 0,
        "rental_rows_in_requested_period": 0,
        "return_rows_in_requested_period": 0,
    }

    for path in files:
        logging.info("Processing rental file: %s", path)
        if path.suffix.lower() != ".csv":
            df = read_table(path)
            aggregates = process_rental_frame(df, station_numbers, start_ts, end_exclusive)
            append_rental_aggregates(aggregates, rental_aggregates, return_aggregates, trip_aggregates)
            stats["raw_rental_rows_loaded"] += aggregates["raw_rows"]
            stats["rental_rows_after_cleaning"] += aggregates["clean_rows_before_date_filter"]
            stats["rental_rows_relevant_to_requested_period"] += aggregates["clean_rows"]
            stats["rental_rows_in_requested_period"] += aggregates["rental_rows_in_requested_period"]
            stats["return_rows_in_requested_period"] += aggregates["return_rows_in_requested_period"]
            used_stations.update(aggregates["used_stations"])
            continue

        columns, encoding = read_csv_header(path)
        selected = find_columns(columns, RENTAL_REQUIRED_COLUMNS, path)
        reader = pd.read_csv(
            path,
            encoding=encoding,
            usecols=list(selected.keys()),
            chunksize=chunksize,
            low_memory=False,
        )
        for chunk in reader:
            aggregates = process_rental_frame(
                chunk.rename(columns=selected),
                station_numbers,
                start_ts,
                end_exclusive,
            )
            append_rental_aggregates(aggregates, rental_aggregates, return_aggregates, trip_aggregates)
            stats["raw_rental_rows_loaded"] += aggregates["raw_rows"]
            stats["rental_rows_after_cleaning"] += aggregates["clean_rows_before_date_filter"]
            stats["rental_rows_relevant_to_requested_period"] += aggregates["clean_rows"]
            stats["rental_rows_in_requested_period"] += aggregates["rental_rows_in_requested_period"]
            stats["return_rows_in_requested_period"] += aggregates["return_rows_in_requested_period"]
            used_stations.update(aggregates["used_stations"])

    rental_counts = combine_count_aggregates(rental_aggregates, "rental_count")
    return_counts = combine_count_aggregates(return_aggregates, "return_count")
    trip_stats = combine_trip_aggregates(trip_aggregates)
    return rental_counts, return_counts, trip_stats, stats, used_stations


def process_rental_frame(
    df: pd.DataFrame,
    station_numbers: set[int],
    start_ts: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> dict[str, object]:
    if set(RENTAL_REQUIRED_COLUMNS.values()).issubset(df.columns):
        rental = df[list(RENTAL_REQUIRED_COLUMNS.values())].copy()
    else:
        selected = find_columns(df.columns, RENTAL_REQUIRED_COLUMNS, Path("<dataframe>"))
        rental = df.rename(columns=selected)[list(RENTAL_REQUIRED_COLUMNS.values())].copy()

    raw_rows = len(rental)
    rental["rental_datetime"] = pd.to_datetime(rental["rental_datetime"], errors="coerce")
    rental["return_datetime"] = pd.to_datetime(rental["return_datetime"], errors="coerce")
    rental["rental_station_number"] = clean_station_number(rental["rental_station_number"])
    rental["return_station_number"] = clean_station_number(rental["return_station_number"])
    rental["duration_min"] = pd.to_numeric(rental["duration_min"], errors="coerce")
    rental["distance_m"] = pd.to_numeric(rental["distance_m"], errors="coerce")

    rental = rental.dropna(
        subset=[
            "rental_datetime",
            "return_datetime",
            "rental_station_number",
            "return_station_number",
            "duration_min",
            "distance_m",
        ]
    )
    rental["rental_station_number"] = rental["rental_station_number"].astype("int64")
    rental["return_station_number"] = rental["return_station_number"].astype("int64")
    rental = rental[
        rental["rental_station_number"].isin(station_numbers)
        & rental["return_station_number"].isin(station_numbers)
    ].copy()
    clean_rows_before_date_filter = len(rental)
    rental_start_mask = (rental["rental_datetime"] >= start_ts) & (
        rental["rental_datetime"] < end_exclusive
    )
    return_mask = (rental["return_datetime"] >= start_ts) & (
        rental["return_datetime"] < end_exclusive
    )
    relevant_mask = rental_start_mask | return_mask
    rental_rows_in_requested_period = int(rental_start_mask.sum())
    return_rows_in_requested_period = int(return_mask.sum())
    rental = rental[relevant_mask].copy()

    if rental.empty:
        return {
            "raw_rows": raw_rows,
            "clean_rows": 0,
            "clean_rows_before_date_filter": clean_rows_before_date_filter,
            "rental_rows_in_requested_period": rental_rows_in_requested_period,
            "return_rows_in_requested_period": return_rows_in_requested_period,
            "rental_counts": empty_count_frame("rental_count"),
            "return_counts": empty_count_frame("return_count"),
            "trip_stats": empty_trip_frame(),
            "used_stations": set(),
        }

    rental["rental_bin"] = rental["rental_datetime"].dt.floor("30min")
    rental["return_bin"] = rental["return_datetime"].dt.floor("30min")
    rental_start_mask = (rental["rental_datetime"] >= start_ts) & (
        rental["rental_datetime"] < end_exclusive
    )
    return_mask = (rental["return_datetime"] >= start_ts) & (
        rental["return_datetime"] < end_exclusive
    )
    rental_start_rows = rental[rental_start_mask]
    return_rows = rental[return_mask]

    if rental_start_rows.empty:
        rental_counts = empty_count_frame("rental_count")
        trip_stats = empty_trip_frame()
    else:
        rental_counts = (
            rental_start_rows.groupby(["rental_bin", "rental_station_number"], observed=True)
            .size()
            .reset_index(name="rental_count")
            .rename(columns={"rental_bin": "timestamp", "rental_station_number": "station_number"})
        )
        trip_stats = (
            rental_start_rows.groupby(["rental_bin", "rental_station_number"], observed=True)
            .agg(
                duration_sum=("duration_min", "sum"),
                distance_sum=("distance_m", "sum"),
                trip_count=("duration_min", "size"),
            )
            .reset_index()
            .rename(columns={"rental_bin": "timestamp", "rental_station_number": "station_number"})
        )
    if return_rows.empty:
        return_counts = empty_count_frame("return_count")
    else:
        return_counts = (
            return_rows.groupby(["return_bin", "return_station_number"], observed=True)
            .size()
            .reset_index(name="return_count")
            .rename(columns={"return_bin": "timestamp", "return_station_number": "station_number"})
        )

    used = set(rental_start_rows["rental_station_number"].unique()).union(
        return_rows["return_station_number"].unique()
    )
    return {
        "raw_rows": raw_rows,
        "clean_rows": len(rental),
        "clean_rows_before_date_filter": clean_rows_before_date_filter,
        "rental_rows_in_requested_period": rental_rows_in_requested_period,
        "return_rows_in_requested_period": return_rows_in_requested_period,
        "rental_counts": rental_counts,
        "return_counts": return_counts,
        "trip_stats": trip_stats,
        "used_stations": used,
    }


def append_rental_aggregates(
    aggregates: dict[str, object],
    rental_aggregates: list[pd.DataFrame],
    return_aggregates: list[pd.DataFrame],
    trip_aggregates: list[pd.DataFrame],
) -> None:
    rental_aggregates.append(aggregates["rental_counts"])
    return_aggregates.append(aggregates["return_counts"])
    trip_aggregates.append(aggregates["trip_stats"])


def empty_count_frame(column: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.Series(dtype="datetime64[ns]"),
            "station_number": pd.Series(dtype="int64"),
            column: pd.Series(dtype="int64"),
        }
    )


def empty_trip_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.Series(dtype="datetime64[ns]"),
            "station_number": pd.Series(dtype="int64"),
            "duration_sum": pd.Series(dtype="float64"),
            "distance_sum": pd.Series(dtype="float64"),
            "trip_count": pd.Series(dtype="int64"),
        }
    )


def combine_count_aggregates(frames: list[pd.DataFrame], column: str) -> pd.DataFrame:
    if not frames:
        return empty_count_frame(column)
    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        return empty_count_frame(column)
    return (
        combined.groupby(["timestamp", "station_number"], as_index=False)[column]
        .sum()
        .sort_values(["timestamp", "station_number"])
    )


def combine_trip_aggregates(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return empty_trip_frame()
    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        return empty_trip_frame()
    combined = combined.groupby(["timestamp", "station_number"], as_index=False)[
        ["duration_sum", "distance_sum", "trip_count"]
    ].sum()
    combined["avg_duration_min"] = combined["duration_sum"] / combined["trip_count"]
    combined["avg_distance_m"] = combined["distance_sum"] / combined["trip_count"]
    return combined[["timestamp", "station_number", "avg_duration_min", "avg_distance_m"]]


def get_korean_holidays(years: Iterable[int]) -> set[pd.Timestamp]:
    try:
        import holidays
    except ImportError as exc:
        raise ImportError(
            "The holidays package is required for Korean public holidays. "
            "Install it with: pip install holidays"
        ) from exc
    return {pd.Timestamp(day) for day in holidays.country_holidays("KR", years=sorted(set(years))).keys()}


def add_time_features(timestamps: pd.DataFrame) -> pd.DataFrame:
    df = timestamps.copy()
    timestamp = df["timestamp"]
    holidays_kr = get_korean_holidays(timestamp.dt.year.unique().tolist())
    df["hour"] = timestamp.dt.hour.astype("int8")
    df["minute"] = timestamp.dt.minute.astype("int8")
    df["day_of_week"] = timestamp.dt.dayofweek.astype("int8")
    df["month"] = timestamp.dt.month.astype("int8")
    df["is_weekend"] = timestamp.dt.dayofweek.isin([5, 6]).astype("int8")
    df["is_holiday"] = timestamp.dt.normalize().isin(holidays_kr).astype("int8")
    return df


def write_panel_in_batches(
    output_path: Path,
    timestamps: pd.DataFrame,
    station_metadata: pd.DataFrame,
    rental_counts: pd.DataFrame,
    return_counts: pd.DataFrame,
    trip_stats: pd.DataFrame,
    weather_30min: pd.DataFrame,
    station_batch_size: int,
) -> tuple[int, list[str], dict[str, int]]:
    if output_path.exists():
        output_path.unlink()

    weather_time = add_time_features(weather_30min)
    station_numbers = station_metadata["station_number"].to_numpy()
    writer: pq.ParquetWriter | None = None
    total_rows = 0
    missing_counts = {column: 0 for column in PANEL_COLUMNS}

    try:
        for start in range(0, len(station_numbers), station_batch_size):
            batch_numbers = station_numbers[start : start + station_batch_size]
            logging.info(
                "Writing station_time_panel batch: stations %s-%s of %s",
                start + 1,
                min(start + station_batch_size, len(station_numbers)),
                len(station_numbers),
            )
            panel = pd.MultiIndex.from_product(
                [timestamps["timestamp"], batch_numbers],
                names=["timestamp", "station_number"],
            ).to_frame(index=False)

            panel = panel.merge(rental_counts, on=["timestamp", "station_number"], how="left")
            panel = panel.merge(return_counts, on=["timestamp", "station_number"], how="left")
            panel = panel.merge(trip_stats, on=["timestamp", "station_number"], how="left")
            panel = panel.merge(weather_time, on="timestamp", how="left")
            panel = panel.merge(
                station_metadata[
                    [
                        "station_number",
                        "latitude",
                        "longitude",
                        "district",
                        "dock_count_raw",
                        "operation_type",
                    ]
                ],
                on="station_number",
                how="left",
            )

            count_columns = ["rental_count", "return_count"]
            panel[count_columns] = panel[count_columns].fillna(0).astype("int32")
            panel["net_demand"] = panel["rental_count"] - panel["return_count"]
            panel[["avg_duration_min", "avg_distance_m"]] = panel[
                ["avg_duration_min", "avg_distance_m"]
            ].fillna(0.0)
            panel = panel[PANEL_COLUMNS]

            batch_missing = panel.isna().sum()
            for column in PANEL_COLUMNS:
                missing_counts[column] += int(batch_missing[column])

            table = pa.Table.from_pandas(panel, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
            total_rows += len(panel)
    finally:
        if writer is not None:
            writer.close()

    return total_rows, PANEL_COLUMNS, missing_counts


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def build_feature_columns() -> dict[str, list[str]]:
    return {
        "target_candidates": ["rental_count", "return_count", "net_demand"],
        "time_features": ["hour", "minute", "day_of_week", "month", "is_weekend", "is_holiday"],
        "weather_features": ["temperature", "wind_speed", "rainfall", "humidity"],
        "station_static_features": [
            "latitude",
            "longitude",
            "district",
            "dock_count_raw",
            "operation_type",
        ],
        "trip_aggregate_features": ["avg_duration_min", "avg_distance_m"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess Seoul public bike rentals, weather, and station metadata."
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--rental-dir", type=Path, default=Path("data/raw/rentals"))
    parser.add_argument("--weather-dir", type=Path, default=Path("data/raw/weather"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/preprocessed/2025"))
    parser.add_argument("--station-output-dir", type=Path, default=Path("data/preprocessed/station"))
    parser.add_argument("--start-date", type=str, default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", type=str, default=DEFAULT_END_DATE)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace known preprocessing output files if they already exist.",
    )
    parser.add_argument(
        "--rebuild-station",
        action="store_true",
        help="Rebuild shared station outputs from raw station metadata.",
    )
    parser.add_argument("--rental-chunksize", type=int, default=1_000_000)
    parser.add_argument("--station-batch-size", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    validate_output_dir(args.output_dir, args.overwrite)
    reuse_station_outputs = validate_station_output_dir(
        args.station_output_dir,
        args.rebuild_station,
    )

    start_ts, end_exclusive, panel_end_ts, timestamp_grid = parse_date_range(
        args.start_date,
        args.end_date,
    )
    timestamps = pd.DataFrame({"timestamp": timestamp_grid})
    years = sorted({int(year) for year in timestamps["timestamp"].dt.year.unique().tolist()})
    logging.info("Requested date range: %s <= timestamp < %s", start_ts, end_exclusive)

    if reuse_station_outputs:
        station_metadata = load_shared_station_metadata(args.station_output_dir)
        logging.info("Loaded shared station outputs from %s", args.station_output_dir)
    else:
        station_metadata = load_station_metadata(args.raw_dir)
        write_shared_station_metadata(args.station_output_dir, station_metadata)
        logging.info("Wrote shared station outputs to %s", args.station_output_dir)
    station_count = len(station_metadata)
    logging.info("Stations available for panel: %s", station_count)

    weather_30min, weather_rows_loaded = load_weather(
        args.weather_dir,
        timestamp_grid,
        start_ts,
        end_exclusive,
    )
    weather_path = args.output_dir / "weather_30min.parquet"
    weather_30min.to_parquet(weather_path, index=False)
    logging.info("Weather rows loaded: %s", weather_rows_loaded)
    logging.info("Weather 30-minute rows: %s", len(weather_30min))

    rental_counts, return_counts, trip_stats, rental_stats, used_stations = process_rentals(
        args.rental_dir,
        set(station_metadata["station_number"].tolist()),
        args.rental_chunksize,
        start_ts,
        end_exclusive,
    )
    rental_rows_dropped = (
        rental_stats["raw_rental_rows_loaded"] - rental_stats["rental_rows_after_cleaning"]
    )
    logging.info("Raw rental rows loaded: %s", rental_stats["raw_rental_rows_loaded"])
    logging.info("Rental rows after cleaning: %s", rental_stats["rental_rows_after_cleaning"])
    logging.info(
        "Rental rows relevant to requested period: %s",
        rental_stats["rental_rows_relevant_to_requested_period"],
    )
    logging.info("Rental rows dropped: %s", rental_rows_dropped)
    logging.info("Rental-start rows in requested period: %s", rental_stats["rental_rows_in_requested_period"])
    logging.info("Return rows in requested period: %s", rental_stats["return_rows_in_requested_period"])

    active_station_count = len(used_stations)
    if active_station_count == 0:
        raise ValueError("No station metadata rows matched cleaned rental data.")
    logging.info("Stations with rental or return activity in requested period: %s", active_station_count)
    write_json(args.output_dir / "feature_columns.json", build_feature_columns())

    panel_rows, panel_columns, missing_values = write_panel_in_batches(
        args.output_dir / "station_time_panel.parquet",
        timestamps,
        station_metadata,
        rental_counts,
        return_counts,
        trip_stats,
        weather_30min,
        args.station_batch_size,
    )
    logging.info("Number of 30-minute timestamps: %s", len(timestamps))
    logging.info("Final station_time_panel shape: (%s, %s)", panel_rows, len(panel_columns))
    logging.info("Missing values by column: %s", missing_values)

    summary = {
        "raw_rental_rows_loaded": int(rental_stats["raw_rental_rows_loaded"]),
        "rental_rows_after_cleaning": int(rental_stats["rental_rows_after_cleaning"]),
        "rental_rows_relevant_to_requested_period": int(
            rental_stats["rental_rows_relevant_to_requested_period"]
        ),
        "rental_rows_dropped": int(rental_rows_dropped),
        "rental_rows_in_requested_period": int(rental_stats["rental_rows_in_requested_period"]),
        "return_rows_in_requested_period": int(rental_stats["return_rows_in_requested_period"]),
        "station_count": int(station_count),
        "active_station_count": int(active_station_count),
        "reused_station_outputs": bool(reuse_station_outputs),
        "station_output_dir": str(args.station_output_dir),
        "time_output_dir": str(args.output_dir),
        "weather_rows_loaded": int(weather_rows_loaded),
        "weather_30min_rows": int(len(weather_30min)),
        "num_timestamps": int(len(timestamps)),
        "num_stations": int(station_count),
        "station_time_panel_rows": int(panel_rows),
        "station_time_panel_columns": panel_columns,
        "missing_values_by_column": missing_values,
        "start_date": str(pd.Timestamp(args.start_date).date()),
        "end_date": str(pd.Timestamp(args.end_date).date()),
        "years": years,
        "start_timestamp": str(start_ts),
        "end_timestamp": str(panel_end_ts),
        "end_exclusive": str(end_exclusive),
    }
    write_json(args.output_dir / "preprocessing_summary.json", summary)
    logging.info("Preprocessing complete. Outputs written to %s", args.output_dir)


if __name__ == "__main__":
    main()
