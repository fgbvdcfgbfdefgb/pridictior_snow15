from __future__ import annotations

import io
import zipfile

from bitcoin_predictor.binance_history import _aggregate_member


def test_aggtrade_csv_is_aggregated_into_second_bars():
    archive_bytes = io.BytesIO()
    csv_content = (
        "aggTradeId,price,quantity,firstTradeId,lastTradeId,transactTime,isBuyerMaker,isBestMatch\n"
        "1,100,0.5,1,1,1704067200000,false,true\n"
        "2,101,0.2,2,2,1704067200000,true,true\n"
        "3,99,1.0,3,3,1704067201000,false,true\n"
    )
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        archive.writestr("sample.csv", csv_content)
    archive_bytes.seek(0)
    with zipfile.ZipFile(archive_bytes) as archive:
        bars = _aggregate_member(archive, "sample.csv", chunksize=2)

    assert len(bars) == 2
    first = bars.iloc[0]
    assert first["open"] == 100.0
    assert first["high"] == 101.0
    assert first["low"] == 100.0
    assert first["close"] == 101.0
    assert first["trade_count"] == 2
    assert first["taker_buy_volume"] == 0.5
