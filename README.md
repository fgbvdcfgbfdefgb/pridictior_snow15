# Bitcoin Price Predictor (BTC/USDT spot)

A research scaffold for **second-by-second historical replay**, delayed-label online learning, multi-GPU candidate training, and a Marimo live dashboard. The default public data source is Binance Spot BTC/USDT because the requested provider name was unspecified. It uses Binance Vision native 1-second kline archives for history and Binance's public Spot WebSocket/REST endpoints for live data.

> **Status of this delivery:** source code, setup scripts, tests, a Snowflake/offline runner, and a separate Marimo app are included. The multi-year dataset and trained weights are **not** bundled: the history is several GB and must be downloaded in monthly ranges, then stored as daily Parquet chunks. No training was run in this workspace because it has no PyTorch/CUDA. This is a research scaffold, not a verified profitable model.

## What it does

- Downloads Binance's native 1-second klines and writes dense UTC **1-second OHLCV + quote volume, trade count, and taker-buy volume** Parquet files. If a second is absent, it is forward-filled at the previous close with zero volume. Daily output files are grouped by month.
- Replays bars in timestamp order. Every forecast targets the log return to the price **1,500 seconds (25 minutes) later**; it is scored only after that future bar arrives. The online path therefore uses delayed labels rather than pretending the reward is known immediately.
- Keeps one CPU-side, deterministic Market Analyser (multi-window feature summaries). Candidate price predictors (`mlp`, `tcn`, `gru`) share the same analyser. The selected model width and analyser window set are chosen **after** hardware discovery.
- Trains one chronological pass (no shuffled epochs), uses a time-ordered holdout, saves every candidate, reports MAE/RMSE/directional accuracy/forecast instability, and selects the lowest composite validation score. `torchrun` + DistributedDataParallel is used when launched with multiple processes/GPUs.
- Includes a live Binance aggregate-trade feed and `notebooks/marimo_live.py`. Its refresh cell replaces the same chart output each time; it does not append a new chart per tick. The dashboard is CMC-inspired (dark price/volume chart), not an official CoinMarketCap integration.

## Important constraints

1. **The dataset is not in Git yet.** From 2020-01-01 through 2026-10-09 there are roughly 214 million one-second slots. Even compressed, the full series is multi-GB. This repository uses daily Parquet chunks and Git LFS tracking, and includes a month-at-a-time commit helper; chunking avoids per-file limits but does not remove LFS storage/bandwidth quotas. Consider object storage/DVC if Git LFS limits are insufficient. Do not put a multi-GB dataset in ordinary Git history.
2. A 1-second OHLCV bar is an aggregation of trades, **not** a full order book or a guarantee that a trade occurred in every second. Binance archive availability and publication delays can make the latest UTC day incomplete. Verify `data/processed/BTCUSDT_1s/manifest.json` before training.
3. No predictor can guarantee stable forecasts or profitable trading. The inference path uses robust loss, a median/EMA stabilizer, and a volatility-aware return cap, but sudden gaps/news can still cause large errors. Paper-trade and backtest with fees/slippage before making any trading decisions. The project does not place orders.
4. The full replay is compute-intensive. The replay can run at maximum speed (`--speed 0`) while preserving market-time order, or at a wall-clock multiplier. Training examples can be sampled with `--sample-stride`; predictions/evaluation in the replay runner remain one per available second.
5. Snowflake must already have the Parquet files staged locally or in an internal stage and the required Python packages available. The Snowflake runner makes **no market-data/network calls**. GPU training in Snowflake requires a Snowflake runtime that actually exposes CUDA; otherwise use CPU or train on MoLab.

## Setup

Python 3.10+ is recommended. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -e ".[train,live,dev]"
```

PyTorch wheels differ by OS/CUDA. If automatic installation does not match the target GPU, install the correct PyTorch build from the official PyTorch installation selector, then install the remaining extras. On MoLab, run the hardware probe first:

```bash
python scripts/detect_hardware.py
```

The size profile is chosen by the **training process** from visible CPU/RAM/GPU resources; this workspace cannot see MoLab's RTX PRO 6000.

## Historical data (monthly 1s-kline source; daily output chunks)

Install Git LFS **before** adding Parquet chunks:

```bash
git lfs install
```

Download a small test month first:

```bash
python scripts/download_binance.py --start 2020-01-01 --end 2020-01-31
```

The default output layout is:

```text
data/processed/BTCUSDT_1s/YYYY-MM/DD.parquet
```

The full range is deliberately not downloaded automatically. Download and review one month at a time (the script skips already existing day files unless `--overwrite` is supplied). Use a completed UTC date as the end; for example, the plan date's latest completed day is `2026-10-09`:

```bash
python scripts/download_binance.py --start 2020-01-01 --end 2026-10-09
```

The downloader rejects today's/future dates so it cannot silently forward-fill unfinished current-day bars. The live feed supplies current-day data. It falls back from a missing monthly Binance Vision archive to daily archives and fails on missing historical source files unless `--skip-missing` is explicitly set. It writes a manifest with source URLs and coverage; inspect it before model training.

To commit already-downloaded chunks in monthly Git commits (requires a configured `origin` and Git LFS):

```bash
bash scripts/commit_data_chunks.sh
```

This commits locally only. Set `PUSH=1` to also push each monthly commit. Git LFS storage and bandwidth limits still apply.

## Train and compare predictor candidates

Training is one-pass and chronological; `--sample-stride 60` means one supervised forecast origin per minute during warm-start training, while the live/replay inference path emits a forecast every second. Each label becomes available only 25 minutes after its forecast origin. No random shuffling or repeated epochs are used.

```bash
python scripts/train.py \
  --data-dir data/processed/BTCUSDT_1s \
  --output-dir models \
  --sample-stride 60
```

For multiple visible NVIDIA GPUs, launch the **same** script under `torchrun`; all ranks participate in DDP for each architecture candidate:

```bash
torchrun --standalone --nproc_per_node=2 scripts/train.py \
  --data-dir data/processed/BTCUSDT_1s \
  --output-dir models \
  --sample-stride 60
```

Use the GPU count reported by `detect_hardware.py`; do not blindly request unavailable GPUs. Artifacts are written to `models/candidates/`, `models/best.pt`, and `models/summary.json`. Every candidate checkpoint includes its feature scaler, hardware profile, architecture, and horizon. Validation is chronological and separated from training by file/time order.

### Historical streaming simulator

```bash
python scripts/replay.py \
  --data-dir data/processed/BTCUSDT_1s \
  --checkpoint models/best.pt \
  --output reports/replay.csv \
  --speed 0
```

`--speed 0` disables wall-clock sleeps but retains timestamp order. `--speed 1` is one historical second per real second; values such as `--speed 60` replay 60 market-seconds per wall-clock second. The CSV contains forecasts, current prices, matured actual 25-minute prices, and errors. `--max-seconds` is useful for a smoke test. Add `--online-update` to fine-tune the best checkpoint once each delayed label matures; use a copy of the checkpoint if you want to preserve the original.

## Live Marimo dashboard on MoLab

After installing dependencies and copying `models/best.pt` to the runtime:

```bash
marimo run notebooks/marimo_live.py --host 0.0.0.0 --port 2718
```

Set `BTC_MODEL_PATH` if the checkpoint is elsewhere. The app attempts to seed **24 hours** of 1-second Binance Spot klines over REST (12 hours for context plus up to 12 hours to warm the longest causal feature), then follows `btcusdt@aggTrade` over WebSocket. The predictor uses the most recent 12 hours; it waits for the full warm-up before forecasting. If the REST endpoint is unavailable or the seed is incomplete, the chart still runs, but the model remains in warm-up. The checkbox for online fine-tuning is off by default. To update weights live, turn it on; matured labels arrive after 25 minutes. A rolling file is written beside the checkpoint as `*_live.pt` after periodic updates.

```bash
export BTC_MODEL_PATH=/path/to/models/best.pt
marimo edit notebooks/marimo_live.py
```

The refresh control reruns one Marimo output cell, which replaces its own Plotly figure. It is not a repeated `display()` loop and does not create a fresh graph output on every tick.

## Snowflake offline training

Stage Parquet chunks to a Snowflake internal stage or make them available in the Python runtime's local filesystem. Install this repository and its Python-only dependencies in the Snowflake environment. Then run:

```bash
python scripts/train_snowflake.py \
  --data-dir /tmp/btc-data/BTCUSDT_1s \
  --output-dir /tmp/btc-models \
  --sample-stride 300
```

The runner does not call Binance or download data. If using `--stage @DB.SCHEMA.BTC_STAGE/path`, provide the Snowflake connection settings required by Snowpark in the runtime, and ensure Snowpark/PyArrow/PyTorch are installed there. For ordinary Snowflake CPU warehouses, force CPU training; multi-GPU DDP requires a GPU-enabled environment and is not implied by this script.

## Repository layout

```text
src/bitcoin_predictor/   features, CPU analyser, predictors, training, replay, live feed
scripts/                 downloader, hardware probe, training, replay, Snowflake runner, chunk commits
notebooks/marimo_live.py separate CMC-inspired live dashboard
snowflake/               offline notes and stage example
 tests/                  feature, history, and stabilizer checks
 data/                   empty until the downloader is run
 models/                 generated checkpoints; never committed by default
```

## Credentials

No GitHub token is required by this code, and no credential is stored in the repository. Authenticate locally with `gh auth login` or your normal Git credential manager. Do not paste tokens into scripts, notebooks, commits, or chat; use the provider's supported secret/environment mechanism.
