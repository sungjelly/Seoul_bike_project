# LSTM Training

This folder contains the reusable training and evaluation entrypoints for the
station-level LSTM experiments.

## Files

- `train_lstm.py`
  - Trains an LSTM model from a YAML config.
  - Handles checkpoints, resume modes, W&B logging, validation, testing, and
    final metric export.
- `evaluate.py`
  - Evaluates a saved checkpoint on one configured split.
- `checkpointing.py`
  - Saves and restores model, optimizer, scheduler, AMP scaler, RNG, and W&B
    run id state.
- `config.py`
  - Loads YAML configs and applies CLI overrides.
- `metrics.py`
  - Converts scaled predictions back to raw counts and computes raw-count
    metrics.
- `utils.py`
  - Shared filesystem, device, seed, JSON, and CLI helpers.

## Dataset Layout

Training uses a version dataset directory, for example:

```text
data/lstm_processed/lstm_v1/
```

The version directory stores window-specific files:

```text
base_data.json
dataset_summary.json
feature_config.json
sample_index_train.npy
sample_index_val.npy
sample_index_<test_split>.npy
splits.json
window_offsets.npy
```

Large arrays are shared through:

```text
data/lstm_processed/base/
```

The loader resolves this through `base_data.json`, so training can still pass:

```bash
--data_dir data/lstm_processed/lstm_v1
```

## Train

Train v1:

```bash
python -m src.training.lstm_training.train_lstm --config configs/models/lstm/lstm_v1.yaml
```

Train v2:

```bash
python -m src.training.lstm_training.train_lstm --config configs/models/lstm/lstm_v2.yaml
```

Common overrides:

```bash
python -m src.training.lstm_training.train_lstm \
  --config configs/models/lstm/lstm_v2.yaml \
  --data_dir data/lstm_processed/lstm_v2 \
  --checkpoint_dir checkpoints/lstm/lstm_v2 \
  --model_dir models/lstm/lstm_v2 \
  --log_dir logs/lstm/lstm_v2 \
  --batch_size 32768 \
  --resume auto \
  --resume_mode auto \
  --wandb_enabled true
```

## Resume Modes

- `--resume auto --resume_mode auto`
  - Uses `last.pt` when available and restores full training state.
- `--resume path/to/best.pt --resume_mode full`
  - Restores model, optimizer, scheduler, AMP scaler, RNG, and counters.
- `--resume path/to/best.pt --resume_mode weights_only`
  - Loads model weights only and starts optimizer/scheduler from the current
    config.

If you deleted W&B runs, old checkpoints may contain stale `wandb_run_id`
values. Clear the checkpoint W&B id or resume weights-only when you want a new
W&B run.

## Evaluate

```bash
python -m src.training.lstm_training.evaluate \
  --config configs/models/lstm/lstm_v1.yaml \
  --checkpoint_path checkpoints/lstm/lstm_v1/best.pt \
  --split test_2025_winter
```

Available split names come from `data/lstm_processed/<version>/splits.json`.

## Colab Versions

The Colab notebooks expose these variables near the top:

```python
DATASET_VERSION = "lstm_v2"
MODEL_VERSION = "lstm_v2"
BASE_DATASET_NAME = "base"
TRAIN_MODULE = "src.training.lstm_training.train_lstm"
CONFIG_PATH = f"configs/models/lstm/{MODEL_VERSION}.yaml"
```

To switch from v1 to v2, change `DATASET_VERSION` and `MODEL_VERSION`. The
notebook derives:

```text
/content/lstm_processed/<DATASET_VERSION>
/content/drive/MyDrive/Seoul_bike_project/checkpoints/lstm/<MODEL_VERSION>
/content/drive/MyDrive/Seoul_bike_project/models/lstm/<MODEL_VERSION>
/content/drive/MyDrive/Seoul_bike_project/logs/lstm/<MODEL_VERSION>
```

## Adding Another Model

`train_lstm.py` currently builds `BaselineLSTM` in `build_model()`. For a
different model, use one of these patterns:

1. Add a new training module, for example:

```text
src/training/lstm_training/train_my_model.py
```

Then set the notebook variable:

```python
TRAIN_MODULE = "src.training.lstm_training.train_my_model"
CONFIG_PATH = "configs/models/lstm/my_model.yaml"
```

2. Or refactor `build_model()` into a model registry keyed by:

```yaml
model:
  architecture: MyModel
```

The second approach is better when several architectures share the same
training loop, batch format, loss, and metrics.
