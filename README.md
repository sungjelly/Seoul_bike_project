# Seoul Bike Demand Prediction

This project predicts Seoul 따릉이 station-level bike demand using historical rental and return data, weather data, time features, and station metadata.

The main model is a baseline LSTM that predicts next-window station demand for rental and return counts.

## Project Structure

```text
configs/        Data and model/training configuration files
data/           Raw, preprocessed, and model-ready data outputs
notebooks/      Colab notebooks grouped by experiment family (`lstm/`, `lstm2/`)
src/            Source code for preprocessing, datasets, models, and training
```

## Setup

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

For GPU training, install the PyTorch build that matches your CUDA environment if the default `torch` package is not appropriate.

## Data Pipeline

Preprocess raw rental, weather, and station metadata files:

```bash
python src/data/preprocess/preprocess_data.py --overwrite
```

This writes shared station files under `data/preprocessed/station` and
date-specific weather/panel files under `data/preprocessed/2025` by default.
Existing shared station files are reused unless you pass `--rebuild-station`.

Build LSTM-ready arrays from the preprocessed panels:

```bash
python src/data/lstm/make_lstm_dataset.py --config configs/data/lstm/lstm_v1_dataset.yaml
```

The dataset config defines the source panels, split date ranges, target columns,
window offsets, and output name. By default it combines:

- `data/preprocessed/2025`
- `data/preprocessed/2024_april_june`

and writes compact model-ready arrays and metadata under
`data/lstm_processed/lstm_v1`.

Dataset versions can share the large base arrays. Build `lstm_v1` first to
create `data/lstm_processed/base`, then build window-only variants such as
`lstm_v2`:

```bash
python src/data/lstm/make_lstm_dataset.py --config configs/data/lstm/lstm_v2_dataset.yaml
```

`lstm_v2` reuses `base` and writes only version-specific window offsets,
sample indexes, and metadata under `data/lstm_processed/lstm_v2`.

The parallel `lstm2` family starts with `tts_lstm2`:

```bash
python -m src.data.lstm2.make_lstm2_dataset --config configs/data/lstm2/tts_lstm2.yaml
```

`lstm2` reuses shared arrays in `data/lstm2_processed/base`. It keeps weather
features inside the LSTM sequence, passes calendar/time features separately to
the final MLP head, and excludes `avg_duration_min` and `avg_distance_m` for now
because those fields may corrupt prediction quality. Those features can be
reintroduced later in a GCN/GNN-specific dataset.

## Training

Train the baseline LSTM with the default config:

```bash
python src/training/lstm_training/train_lstm.py \
  --config configs/models/lstm/lstm_v1.yaml \
  --data_dir data/lstm_processed/lstm_v1
```

For the sparse-window `lstm_v2` dataset:

```bash
python src/training/lstm_training/train_lstm.py --config configs/models/lstm/lstm_v2.yaml
```

Train `tts_lstm2`:

```bash
python -m src.training.lstm2_training.train_lstm2 --config configs/models/lstm2/tts_lstm2.yaml
```

Run the one-batch `lstm2` smoke train:

```bash
python -m src.training.lstm2_training.train_lstm2 --config configs/models/lstm2/tts_lstm2.yaml --smoke_test true --max_epochs 1
```

The config controls data paths, model size, batch size, checkpointing, W&B logging, and evaluation settings.

## Evaluation

Evaluate a trained checkpoint:

```bash
python src/training/lstm_training/evaluate.py \
  --checkpoint_path checkpoints/lstm/lstm_v1/best.pt \
  --data_dir data/lstm_processed/lstm_v1 \
  --split test_2025_winter
```

Run simple raw-count baselines:

```bash
python src/training/naive_training/naive_baseline.py
```

## Main Dependencies

- numpy
- pandas
- pyarrow
- scikit-learn
- matplotlib
- torch
- tqdm
- pyyaml
- wandb
- holidays
- openpyxl
- xlrd
