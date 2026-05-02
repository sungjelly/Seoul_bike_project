# Seoul Bike Demand Prediction

This project predicts Seoul 따릉이 station-level bike demand using historical rental and return data, weather data, time features, and station metadata.

The main model is a baseline LSTM that predicts next-window station demand for rental and return counts.

## Project Structure

```text
configs/        Model and training configuration files
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

This writes cleaned parquet, numpy, and JSON files under `data/preprocessed/2025` by default.

Build LSTM-ready arrays from the preprocessed panel:

```bash
python src/data/lstm_baseline/make_lstm_dataset.py
```

This writes model-ready arrays and split metadata under `data/lstm_baseline`.

## Training

Train the baseline LSTM with the default config:

```bash
python src/training/lstm_training/train_lstm.py --config configs/lstm_baseline.yaml
```

The config controls data paths, model size, batch size, checkpointing, W&B logging, and evaluation settings.

## Evaluation

Evaluate a trained checkpoint:

```bash
python src/training/lstm_training/evaluate.py --checkpoint_path checkpoints/lstm_baseline/best.pt --split test
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

