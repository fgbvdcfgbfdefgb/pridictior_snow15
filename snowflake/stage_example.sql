-- Example only. Run as a role authorized to create stages and upload files.
CREATE STAGE IF NOT EXISTS BTC_DATA_STAGE
  DIRECTORY = (ENABLE = TRUE);

-- Upload Parquet chunks from a trusted client with SnowSQL/Snowflake CLI, for example:
-- PUT file:///local/path/BTCUSDT_1s/2020-01/*.parquet @BTC_DATA_STAGE/BTCUSDT_1s/2020-01/
-- AUTO_COMPRESS = FALSE OVERWRITE = FALSE;
-- Then run scripts/train_snowflake.py against @<DATABASE>.<SCHEMA>.BTC_DATA_STAGE/BTCUSDT_1s/.
