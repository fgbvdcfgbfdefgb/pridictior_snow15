"""Shared project dimensions and market-time constants."""

SYMBOL = "BTCUSDT"
BAR_SECONDS = 1
HORIZON_SECONDS = 25 * 60
CONTEXT_SECONDS = 12 * 60 * 60
TARGET_SCALE = 100.0  # model target is log-return expressed in percentage points

RAW_BAR_COLUMNS = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trade_count",
    "taker_buy_volume",
)
