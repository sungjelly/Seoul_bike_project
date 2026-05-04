# Naive Baselines

`naive_baseline.py` evaluates raw-count baselines against the generated LSTM
dataset splits.

The script supports the shared dataset layout:

```text
data/lstm_processed/base/
data/lstm_processed/lstm_v1/
data/lstm_processed/lstm_v2/
```

It reads `targets_raw.npy` from the shared base directory and reads
`sample_index_<split>.npy`, `splits.json`, and `source_boundaries.json` from the
selected dataset version.

## W&B-Only Logging

By default, results are printed and logged to W&B. Local JSON/CSV files are not
written unless `--save-local true` is passed. Each naive baseline is logged as
its own W&B run.

```bash
python -m src.training.naive_training.naive_baseline \
  --data-dir data/lstm_processed/lstm_v1 \
  --splits val test_2025_winter test_2024_april_june \
  --wandb-enabled true
```

Useful overrides:

```bash
python -m src.training.naive_training.naive_baseline \
  --data-dir data/lstm_processed/lstm_v2 \
  --wandb-name naive_baseline_lstm_v2 \
  --wandb-group lstm_v2 \
  --save-local false
```

Default run names are just the baseline names:

```text
<baseline_name>
```

For `lstm_v1`, this creates six runs grouped under `lstm_v1`:

```text
zero
previous_window
same_time_yesterday
same_time_last_week
recent_mean_4h
recent_mean_24h
```

Metrics inside each run are logged as:

```text
<split>/<metric>
```

Example:

```text
val/total_mae
test_2025_winter/return_rmse
test_2025_winter/net_demand_mae
```

The history-based baselines are filtered so their lookback windows do not cross
source boundaries.
