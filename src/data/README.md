# Data Processing Code

This folder contains dataset preparation code only. It does not define models,
train models, create graph edges, or run experiments.

## Layout

- `preprocess/preprocess_data.py`
  - Builds the model-agnostic preprocessed base data from raw rentals, weather,
    and station metadata.
  - Main output: `data/preprocessed/station_time_panel.parquet`.

- `lstm_baseline/make_lstm_dataset.py`
  - Builds the supervised baseline LSTM dataset from `data/preprocessed`.
  - Saves base arrays only, not materialized sequence windows.
  - Main output: `data/lstm_baseline/`.

- `lstm_baseline/lstm_dataset.py`
  - Defines a lazy PyTorch Dataset that slices LSTM windows on demand from
    `data/lstm_baseline`.

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

Partial 2024 example:

```bash
python src/data/preprocess/preprocess_data.py \
  --start-date 2024-04-01 \
  --end-date 2024-06-30 \
  --output-dir data/preprocessed/2024_april_june
```

Then build the LSTM baseline dataset:

```bash
python src/data/lstm_baseline/make_lstm_dataset.py \
  --preprocessed_dir data/preprocessed/2025 \
  --output_dir data/lstm_baseline \
  --sequence_length 8 \
  --horizon 1
```
