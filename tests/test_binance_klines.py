from __future__ import annotations

import io
import zipfile

from bitcoin_predictor.binance_klines import _aggregate_kline_member


def test_native_one_second_klines_are_parsed():
    archive_bytes = io.BytesIO()
    csv_content = (
        "open_time,open,high,low,close,volume,close_time,quote_volume,trade_count,taker_buy_volume,taker_buy_quote_volume,ignore\n"
        "1704067200000,100,101,99,100.5,2,1704067200999,201,4,1.2,120.5,0\n"
        "1704067202000,100.5,102,100,101.5,3,1704067202999,304,5,2.0,202,0\n"
    )
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        archive.writestr("BTCUSDT-1s-sample.csv", csv_content)
    archive_bytes.seek(0)
    with zipfile.ZipFile(archive_bytes) as archive:
        bars = _aggregate_kline_member(archive, "BTCUSDT-1s-sample.csv")

    assert len(bars) == 2
    assert bars.iloc[0]["timestamp"].isoformat() == "2024-01-01T00:00:00+00:00"
    assert bars.iloc[0]["close"] == 100.5
    assert bars.iloc[0]["trade_count"] == 4
    assert bars.iloc[1]["taker_buy_volume"] == 2.0
