"""Causal second-level features, a CPU Market Analyser, and live ring buffers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .constants import CONTEXT_SECONDS, HORIZON_SECONDS

RETURN_LAGS = (1, 5, 10, 30, 60, 300, 900, 3600, 21600, 43200)
VOL_WINDOWS = (10, 30, 60, 300, 900, 3600, 21600, 43200)
FEATURE_NAMES = tuple(
    [f"log_return_{lag}s" for lag in RETURN_LAGS]
    + [
        "range_fraction",
        "log_volume",
        "log_quote_volume",
        "log_trade_count",
        "taker_imbalance",
    ]
    + [f"realized_vol_{window}s" for window in VOL_WINDOWS]
    + ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos"]
)
FEATURE_DIM = len(FEATURE_NAMES)
MAX_FEATURE_LOOKBACK_SECONDS = max(RETURN_LAGS)


def _timestamps(frame: pd.DataFrame) -> pd.Series:
    if "timestamp" in frame.columns:
        return pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if isinstance(frame.index, pd.DatetimeIndex):
        return pd.Series(pd.to_datetime(frame.index, utc=True), index=frame.index)
    raise ValueError("Bars need a UTC 'timestamp' column or DatetimeIndex")


def build_features(bars: pd.DataFrame) -> np.ndarray:
    """Create causal feature rows from a dense or sparse OHLCV frame.

    The function never reads a future row: every return, rolling statistic, and
    timestamp feature is available at the end of that bar.
    """
    if bars.empty:
        return np.empty((0, FEATURE_DIM), dtype=np.float32)
    required = {"close", "high", "low", "volume"}
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"Missing bar columns: {sorted(missing)}")

    close = pd.to_numeric(bars["close"], errors="coerce").astype("float64").clip(lower=1e-12)
    high = pd.to_numeric(bars["high"], errors="coerce").astype("float64").fillna(close)
    low = pd.to_numeric(bars["low"], errors="coerce").astype("float64").fillna(close)
    volume = pd.to_numeric(bars["volume"], errors="coerce").fillna(0.0).clip(lower=0.0)
    quote_volume = pd.to_numeric(
        bars["quote_volume"] if "quote_volume" in bars else close * volume,
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0)
    trade_count = pd.to_numeric(
        bars["trade_count"] if "trade_count" in bars else pd.Series(0.0, index=bars.index),
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0)
    taker_buy = pd.to_numeric(
        bars["taker_buy_volume"] if "taker_buy_volume" in bars else pd.Series(0.0, index=bars.index),
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0)

    log_close = np.log(close)
    ret_1 = log_close.diff().fillna(0.0)
    columns: list[pd.Series] = []
    for lag in RETURN_LAGS:
        columns.append((log_close - log_close.shift(lag)).fillna(0.0))
    columns.extend(
        [
            ((high - low) / close).replace([np.inf, -np.inf], 0.0).fillna(0.0),
            np.log1p(volume),
            np.log1p(quote_volume),
            np.log1p(trade_count),
            ((2.0 * taker_buy / volume.replace(0.0, np.nan)) - 1.0).fillna(0.0).clip(-1.0, 1.0),
        ]
    )
    squared_returns = ret_1.pow(2)
    for window in VOL_WINDOWS:
        realized = squared_returns.rolling(window=window, min_periods=1).sum().clip(lower=0.0)
        columns.append(np.sqrt(realized))

    ts = _timestamps(bars)
    hour_fraction = (
        ts.dt.hour.fillna(0).astype("float64")
        + ts.dt.minute.fillna(0).astype("float64") / 60.0
        + ts.dt.second.fillna(0).astype("float64") / 3600.0
    )
    weekday_fraction = ts.dt.dayofweek.fillna(0).astype("float64") / 7.0
    columns.extend(
        [
            np.sin(2.0 * np.pi * hour_fraction / 24.0),
            np.cos(2.0 * np.pi * hour_fraction / 24.0),
            np.sin(2.0 * np.pi * weekday_fraction),
            np.cos(2.0 * np.pi * weekday_fraction),
        ]
    )
    result = np.column_stack([np.asarray(col, dtype=np.float64) for col in columns])
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


@dataclass
class FeatureScaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, rows: np.ndarray) -> "FeatureScaler":
        values = np.asarray(rows, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError("Scaler fit requires a non-empty 2D feature matrix")
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale = np.where(scale < 1e-6, 1.0, scale)
        return cls(mean.astype(np.float32), scale.astype(np.float32))

    def transform(self, rows: np.ndarray) -> np.ndarray:
        values = np.asarray(rows, dtype=np.float32)
        result = (values - self.mean) / self.scale
        return np.nan_to_num(np.clip(result, -12.0, 12.0), nan=0.0, posinf=12.0, neginf=-12.0).astype(
            np.float32
        )

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.astype(float).tolist(), "scale": self.scale.astype(float).tolist()}

    @classmethod
    def from_dict(cls, value: dict[str, Iterable[float]]) -> "FeatureScaler":
        return cls(np.asarray(value["mean"], dtype=np.float32), np.asarray(value["scale"], dtype=np.float32))


class MarketAnalyzer:
    """CPU-side, deterministic multi-window summary shared by all predictors.

    More capable hosts get more rolling windows/signals from `ModelProfile`; no
    separate analyser is fit for each predictor candidate.
    """

    def __init__(self, windows: Iterable[int] = (60, 300, 3600)) -> None:
        self.windows = tuple(sorted(set(int(w) for w in windows if int(w) > 0)))
        if not self.windows:
            raise ValueError("MarketAnalyzer needs at least one positive window")
        self.output_dim = FEATURE_DIM * (1 + 2 * len(self.windows))

    def analyze(self, normalized_history: np.ndarray) -> np.ndarray:
        values = np.asarray(normalized_history, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != FEATURE_DIM:
            raise ValueError(f"Expected non-empty [time, {FEATURE_DIM}] feature history")
        parts = [values[-1]]
        for window in self.windows:
            segment = values[-min(window, len(values)) :]
            parts.extend((segment.mean(axis=0), segment.std(axis=0)))
        return np.concatenate(parts).astype(np.float32, copy=False)


class RollingFeatureEngine:
    """Incremental causal feature engine for a live one-second bar stream."""

    def __init__(self, context_seconds: int = CONTEXT_SECONDS, horizon_seconds: int = HORIZON_SECONDS) -> None:
        self.context_seconds = int(context_seconds)
        self.horizon_seconds = int(horizon_seconds)
        self._closes: deque[float] = deque(maxlen=max(RETURN_LAGS) + 1)
        self._return_squares: dict[int, deque[float]] = {w: deque() for w in VOL_WINDOWS}
        self._return_square_sums = {w: 0.0 for w in VOL_WINDOWS}
        self._features: deque[np.ndarray] = deque(maxlen=self.context_seconds + self.horizon_seconds + 1)
        self.rows_seen = 0

    def update(self, bar: dict) -> np.ndarray:
        close = max(float(bar["close"]), 1e-12)
        log_close = float(np.log(close))
        returns: list[float] = []
        for lag in RETURN_LAGS:
            if len(self._closes) >= lag:
                returns.append(log_close - float(np.log(max(self._closes[-lag], 1e-12))))
            else:
                returns.append(0.0)
        ret_1 = returns[0]
        for window in VOL_WINDOWS:
            queue = self._return_squares[window]
            if len(queue) >= window:
                self._return_square_sums[window] -= queue.popleft()
            square = ret_1 * ret_1
            queue.append(square)
            self._return_square_sums[window] += square

        self._closes.append(close)
        high = float(bar.get("high", close))
        low = float(bar.get("low", close))
        volume = max(float(bar.get("volume", 0.0)), 0.0)
        quote_volume = max(float(bar.get("quote_volume", close * volume)), 0.0)
        trade_count = max(float(bar.get("trade_count", 0.0)), 0.0)
        taker_buy = max(float(bar.get("taker_buy_volume", 0.0)), 0.0)
        imbalance = (2.0 * taker_buy / volume - 1.0) if volume > 0.0 else 0.0
        imbalance = float(np.clip(imbalance, -1.0, 1.0))

        timestamp = pd.Timestamp(bar["timestamp"])
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        hour_fraction = timestamp.hour + timestamp.minute / 60.0 + timestamp.second / 3600.0
        weekday_fraction = timestamp.dayofweek / 7.0
        row = np.asarray(
            returns
            + [
                (high - low) / close,
                float(np.log1p(volume)),
                float(np.log1p(quote_volume)),
                float(np.log1p(trade_count)),
                imbalance,
            ]
            + [float(np.sqrt(max(self._return_square_sums[w], 0.0))) for w in VOL_WINDOWS]
            + [
                float(np.sin(2.0 * np.pi * hour_fraction / 24.0)),
                float(np.cos(2.0 * np.pi * hour_fraction / 24.0)),
                float(np.sin(2.0 * np.pi * weekday_fraction)),
                float(np.cos(2.0 * np.pi * weekday_fraction)),
            ],
            dtype=np.float32,
        )
        self._features.append(row)
        self.rows_seen += 1
        return row

    def history_array(self, length: int | None = None, delay: int = 0) -> np.ndarray | None:
        """Return a feature window ending `delay` bars before the latest row."""
        if not self._features:
            return None
        matrix = np.asarray(list(self._features), dtype=np.float32)
        end = len(matrix) - int(delay)
        if length is None:
            start = 0
        else:
            start = end - int(length)
        if start < 0 or end <= start:
            return None
        return matrix[start:end]


def make_model_input(
    normalized_history: np.ndarray,
    context_seconds: int = CONTEXT_SECONDS,
    recent_seconds: int = 1500,
    coarse_bins: int = 512,
) -> np.ndarray:
    """Reduce a full 12-hour, 1-second window to a multi-resolution sequence.

    Every second is available to this function. The recent 25 minutes remain at
    1-second resolution; the older history is pooled into equal-time bins so
    long-context models remain practical.
    """
    values = np.asarray(normalized_history, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected [time, {FEATURE_DIM}] history")
    if len(values) < context_seconds:
        raise ValueError(f"Need {context_seconds} one-second rows, got {len(values)}")
    values = values[-context_seconds:]
    recent_seconds = min(int(recent_seconds), len(values) - 1)
    old = values[:-recent_seconds]
    edges = np.linspace(0, len(old), int(coarse_bins) + 1, dtype=np.int64)
    prefix = np.vstack((np.zeros((1, old.shape[1]), dtype=np.float64), np.cumsum(old, axis=0, dtype=np.float64)))
    widths = np.maximum(edges[1:] - edges[:-1], 1)
    coarse = (prefix[edges[1:]] - prefix[edges[:-1]]) / widths[:, None]
    result = np.concatenate((coarse.astype(np.float32), values[-recent_seconds:]), axis=0)
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
