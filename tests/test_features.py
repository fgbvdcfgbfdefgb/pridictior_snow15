from __future__ import annotations

import numpy as np
import pandas as pd

from bitcoin_predictor.constants import CONTEXT_SECONDS
from bitcoin_predictor.features import (
    FEATURE_DIM,
    FeatureScaler,
    MarketAnalyzer,
    RollingFeatureEngine,
    build_features,
    make_model_input,
)


def sample_bars(count: int = 2500) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0, 0.00005, size=count)
    close = 60_000 * np.exp(np.cumsum(returns))
    timestamp = pd.date_range("2024-01-01", periods=count, freq="1s", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "open": close,
            "high": close * 1.0001,
            "low": close * 0.9999,
            "close": close,
            "volume": rng.uniform(0.0, 2.0, size=count),
            "quote_volume": rng.uniform(0.0, 120_000.0, size=count),
            "trade_count": rng.integers(0, 10, size=count),
            "taker_buy_volume": rng.uniform(0.0, 1.0, size=count),
        }
    )


def test_batch_and_incremental_features_agree():
    bars = sample_bars()
    batch = build_features(bars)
    engine = RollingFeatureEngine(context_seconds=300, horizon_seconds=30)
    incremental = np.stack([engine.update(row) for row in bars.to_dict(orient="records")])
    assert batch.shape == incremental.shape == (len(bars), FEATURE_DIM)
    np.testing.assert_allclose(batch[-100:], incremental[-100:], rtol=2e-4, atol=2e-6)


def test_scaler_and_analyzer_dimensions():
    raw = build_features(sample_bars())
    scaler = FeatureScaler.fit(raw)
    normalized = scaler.transform(raw)
    assert np.isfinite(normalized).all()
    analyzer = MarketAnalyzer((60, 300, 900))
    signals = analyzer.analyze(normalized[-500:])
    assert signals.shape == (analyzer.output_dim,)
    assert np.isfinite(signals).all()


def test_multiresolution_input_uses_full_context():
    rng = np.random.default_rng(1)
    context = rng.normal(size=(CONTEXT_SECONDS, FEATURE_DIM)).astype(np.float32)
    reduced = make_model_input(context)
    assert reduced.shape == (512 + 1500, FEATURE_DIM)
    np.testing.assert_allclose(reduced[-1500:], context[-1500:])
