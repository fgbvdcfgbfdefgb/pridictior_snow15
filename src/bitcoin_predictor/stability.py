"""Forecast stabilization helpers (safeguards, not a guarantee of accuracy)."""

from __future__ import annotations

from collections import deque

import numpy as np


class PredictionStabilizer:
    def __init__(
        self,
        alpha: float = 0.25,
        median_window: int = 5,
        volatility_multiplier: float = 6.0,
        floor_cap: float = 0.003,
        hard_cap: float = 0.25,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = float(alpha)
        self.volatility_multiplier = float(volatility_multiplier)
        self.floor_cap = float(floor_cap)
        self.hard_cap = float(hard_cap)
        self._recent: deque[float] = deque(maxlen=max(1, int(median_window)))
        self._last: float | None = None

    def update(self, raw_log_return: float, realized_vol_horizon: float = 0.0) -> float:
        """Return a smoothed log-return forecast with volatility-aware clipping."""
        cap = min(self.hard_cap, max(self.floor_cap, self.volatility_multiplier * max(realized_vol_horizon, 0.0)))
        clipped = float(np.clip(raw_log_return, -cap, cap))
        self._recent.append(clipped)
        robust = float(np.median(np.asarray(self._recent, dtype=np.float64)))
        if self._last is None:
            value = robust
        else:
            step_cap = max(self.floor_cap, 2.0 * max(realized_vol_horizon, 0.0))
            robust = float(np.clip(robust, self._last - step_cap, self._last + step_cap))
            value = self.alpha * robust + (1.0 - self.alpha) * self._last
        self._last = value
        return value

    def reset(self) -> None:
        self._recent.clear()
        self._last = None
