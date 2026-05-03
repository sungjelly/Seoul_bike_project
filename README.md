# Seoul Bike Demand Prediction

This project predicts Seoul 따릉이 station-level bike demand using historical rental and return data, weather data, time features, and station metadata.

The main model is a baseline LSTM that predicts next-window station demand for rental and return counts.

## Project Structure

```text
configs/        Data and model/training configuration files
data/           Raw, preprocessed, and model-ready data outputs
notebooks/      Colab notebooks for training and evaluation
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
python src/data/lstm_baseline/make_lstm_dataset.py --config configs/data/lstm/lstm_v1_dataset.yaml
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
python src/data/lstm_baseline/make_lstm_dataset.py --config configs/data/lstm/lstm_v2_dataset.yaml
```

`lstm_v2` reuses `base` and writes only version-specific window offsets,
sample indexes, and metadata under `data/lstm_processed/lstm_v2`.

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

The config controls data paths, model size, batch size, checkpointing, W&B logging, and evaluation settings.

## Evaluation

Evaluate a trained checkpoint:

```bash
python src/training/lstm_training/evaluate.py \
  --checkpoint_path checkpoints/lstm_v1/best.pt \
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
