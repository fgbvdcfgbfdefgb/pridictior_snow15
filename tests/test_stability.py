from __future__ import annotations

from bitcoin_predictor.stability import PredictionStabilizer


def test_stabilizer_clips_jumps_and_returns_finite_values():
    smoother = PredictionStabilizer(alpha=0.2, median_window=3, hard_cap=0.05)
    values = [smoother.update(0.0, 0.001), smoother.update(0.5, 0.001), smoother.update(-0.5, 0.001)]
    assert all(abs(value) <= 0.05 for value in values)
    # The one-tick extremes are clipped, median-filtered, and smoothed.
    assert abs(values[1] - values[0]) < 0.01
    assert abs(values[2]) < 0.01
