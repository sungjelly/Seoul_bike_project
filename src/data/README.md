# Data Processing Code

This folder contains dataset preparation code only. It does not define models,
train models, create graph edges, or run experiments.

## Layout

- `preprocess/preprocess_data.py`
  - Builds the model-agnostic preprocessed base data from raw rentals, weather,
    and station metadata.
  - Shared station output: `data/preprocessed/station/`.
  - Main time-range output: `data/preprocessed/<range>/station_time_panel.parquet`.

- `lstm_baseline/make_lstm_dataset.py`
  - Builds the supervised baseline LSTM dataset from one or more
    `data/preprocessed` source panels.
  - Saves base arrays only, not materialized sequence windows.
  - Can write large shared arrays to `data/lstm_processed/<base_dataset_name>/`
    and lightweight window-specific metadata to
    `data/lstm_processed/<dataset_name>/`.

- `lstm_baseline/lstm_dataset.py`
  - Defines a lazy PyTorch Dataset that slices LSTM windows on demand from
    the compact arrays produced by `make_lstm_dataset.py`.

- `lstm_baseline/scaling.py`
  - Reusable transform and inverse-transform helpers.

## Commands

From the project root:

```bash
python src/data/preprocess/preprocess_data.py
```

The preprocessing script can also build standalone datasets for arbitrary date
ranges. It does not append to existing preprocessed outputs.

Full 2025 example:

```bash
python src/data/preprocess/preprocess_data.py \
  --start-date 2025-01-01 \
  --end-date 2025-12-31 \
  --output-dir data/preprocessed/2025 \
  --overwrite
```

Station metadata is written once to `data/preprocessed/station` and reused by
later preprocessing runs. Use `--rebuild-station` only when you intentionally
want to recreate the shared station files from raw metadata.

Partial 2024 example:

```bash
python src/data/preprocess/preprocess_data.py \
  --start-date 2024-04-01 \
  --end-date 2024-06-30 \
  --output-dir data/preprocessed/2024_april_june
```

Then build the LSTM baseline dataset:

```bash
python src/data/lstm_baseline/make_lstm_dataset.py --config configs/lstm_v1_dataset.yaml
```

The v1 config builds shared base arrays in `data/lstm_processed/base` and
version-specific sample indexes in `data/lstm_processed/lstm_v1` from these
configured sources:

- `data/preprocessed/2025`
- `data/preprocessed/2024_april_june`

The config controls `dataset_name`, `output_dir`, source panel paths, split date
ranges, target columns, horizon, explicit LSTM window offsets, and whether large
base arrays should be reused.

Build `lstm_v2` after `lstm_v1` when you only want to change the LSTM window:

```bash
python src/data/lstm_baseline/make_lstm_dataset.py --config configs/lstm_v2_dataset.yaml
```

`configs/lstm_v2_dataset.yaml` sets `reuse_base_arrays: true`, so it reads the
large arrays from `data/lstm_processed/base` and writes only
`lstm_v2`-specific sample indexes, window offsets, and metadata.

CLI overrides are available for path-level changes:

```bash
python src/data/lstm_baseline/make_lstm_dataset.py \
  --config configs/lstm_v1_dataset.yaml \
  --output-dir data/lstm_processed \
  --dataset-name lstm_v1
```

Expected shared base files include:

- `dynamic_features.npy`
- `targets.npy`
- `targets_raw.npy`
- `static_numeric.npy`
- `district_ids.npy`
- `operation_type_ids.npy`
- `station_numbers.npy`
- `timestamps.npy`
- `scalers.json`
- `source_boundaries.json`
- `station_index_map.json`
- `district_vocab.json`
- `operation_type_vocab.json`
- `dataset_summary.json`

Expected version-specific files include:

- `sample_index_<split>.npy`
- `window_offsets.npy`
- `base_data.json`
- `feature_config.json`
- `splits.json`
- `scalers.json`
- `source_boundaries.json`
- `dataset_summary.json`
