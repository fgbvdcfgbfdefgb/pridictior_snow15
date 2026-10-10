from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("torch")
from bitcoin_predictor.training import _join_contiguous


def test_join_contiguous_resets_on_market_time_gap():
    history = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=3, freq="s", tz="UTC")})
    next_day = pd.DataFrame({"timestamp": pd.date_range("2024-01-01 00:00:04", periods=2, freq="s", tz="UTC")})
    joined, old = _join_contiguous(history, next_day)
    assert old == 3
    assert len(joined) == 5
    gap = pd.DataFrame({"timestamp": pd.date_range("2024-01-01 01:00:00", periods=2, freq="s", tz="UTC")})
    joined_gap, old_gap = _join_contiguous(history, gap)
    assert old_gap == 0
    assert len(joined_gap) == 2
