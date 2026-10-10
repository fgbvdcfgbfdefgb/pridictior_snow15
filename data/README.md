# Data directory

Run `python scripts/download_binance.py --start YYYY-MM-DD --end YYYY-MM-DD` to download Binance Vision's native 1-second BTC/USDT klines and create daily Parquet chunks under `processed/BTCUSDT_1s/YYYY-MM/DD.parquet`.

The repository includes the complete 2020-01-01 through 2026-10-09 dataset in Git LFS: 2,474 daily Parquet files and 213,753,600 one-second rows (about 7.2 GB). After cloning, run `git lfs install` and `git lfs pull` to fetch the objects. `processed/BTCUSDT_1s/manifest.json` records the source, range, per-file coverage, and missing dates; the committed manifest reports complete coverage. Raw source archives are temporary unless `--keep-raw` is supplied, and `/data/raw/` is ignored by Git.
