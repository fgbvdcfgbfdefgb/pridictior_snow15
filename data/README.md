# Data directory

Run `python scripts/download_binance.py --start YYYY-MM-DD --end YYYY-MM-DD` to create daily 1-second BTC/USDT Parquet chunks under `processed/BTCUSDT_1s/YYYY-MM/DD.parquet`.

The full dataset is intentionally not included in this source bundle. Set up Git LFS before adding any chunks; see the repository README. Raw source archives are temporary unless `--keep-raw` is supplied, and `/data/raw/` is ignored by Git. `processed/BTCUSDT_1s/manifest.json` records source, range, and missing dates after a download.
