# Snowflake offline training notes

- Upload daily Parquet chunks to an **internal** stage from a trusted client before training (for example with SnowSQL `PUT file://... @DB.SCHEMA.BTC_STAGE/BTCUSDT_1s AUTO_COMPRESS=FALSE`). The full dataset is multi-GB; stage and warehouse capacity must be planned.
- Run `scripts/train_snowflake.py --stage @DB.SCHEMA.BTC_STAGE/BTCUSDT_1s/ --output-dir @local-path --sample-stride 300` in an environment with an active Snowpark session and required Python packages. Or use `--data-dir` if the files are already local.
- The training code reads only staged/local Parquet files; it does not fetch data from Binance or require public internet access. `torch`, `pandas`, `pyarrow`, and the source package must be installed/available in the Snowflake Python runtime.
- Most standard Snowflake warehouses do not expose CUDA. The wrapper defaults to `--device cpu`; multi-GPU training requires a Snowflake GPU runtime and launching the DDP script with an appropriate `torchrun` environment. Otherwise train on a GPU host and use Snowflake for offline analysis/validation.
- Keep credentials in Snowflake-managed secrets/connections, not in SQL files or this repository.
