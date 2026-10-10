"""Binance public feed, 1-second bars, and live/historical prediction runner."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import threading
import time
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import torch
import websocket

from .constants import CONTEXT_SECONDS, HORIZON_SECONDS, SYMBOL, TARGET_SCALE
from .features import (
    FEATURE_NAMES,
    MAX_FEATURE_LOOKBACK_SECONDS,
    FeatureScaler,
    MarketAnalyzer,
    RollingFeatureEngine,
    make_model_input,
)
from .hardware import ModelProfile, choose_model_profile, detect_hardware
from .model_io import load_checkpoint, save_live_checkpoint
from .stability import PredictionStabilizer

LOG = logging.getLogger("bitcoin_predictor.live")
KLINES_URL = "https://api.binance.com/api/v3/klines"
WS_URL = "wss://stream.binance.com:9443/ws/{symbol_lower}@aggTrade"


def fetch_recent_klines(
    symbol: str = SYMBOL,
    lookback_seconds: int = CONTEXT_SECONDS + MAX_FEATURE_LOOKBACK_SECONDS,
    timeout: float = 12.0,
) -> list[dict]:
    """Fetch closed 1-second candles in pages, then fill any empty seconds."""
    now_ms = int(time.time() * 1000)
    last_closed_open_ms = now_ms - (now_ms % 1000) - 1000
    first_open_ms = max(0, last_closed_open_ms - (int(lookback_seconds) - 1) * 1000)
    cursor = first_open_ms
    rows: list[list] = []
    page_count = 0
    while cursor <= last_closed_open_ms and page_count < 100:
        response = requests.get(
            KLINES_URL,
            params={
                "symbol": symbol.upper(),
                "interval": "1s",
                "startTime": cursor,
                "endTime": last_closed_open_ms,
                "limit": 1000,
            },
            timeout=timeout,
            headers={"User-Agent": "btc-second-bars/0.1"},
        )
        response.raise_for_status()
        page = response.json()
        if not page:
            break
        rows.extend(page)
        next_cursor = int(page[-1][0]) + 1000
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        page_count += 1
        if len(page) < 1000:
            break
        time.sleep(0.04)
    if not rows:
        return []

    rows = [row for row in rows if first_open_ms <= int(row[0]) <= last_closed_open_ms]
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime([int(row[0]) for row in rows], unit="ms", utc=True),
            "open": [float(row[1]) for row in rows],
            "high": [float(row[2]) for row in rows],
            "low": [float(row[3]) for row in rows],
            "close": [float(row[4]) for row in rows],
            "volume": [float(row[5]) for row in rows],
            "quote_volume": [float(row[7]) for row in rows],
            "trade_count": [int(row[8]) for row in rows],
            "taker_buy_volume": [float(row[9]) for row in rows],
        }
    ).drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    if frame.empty:
        return []
    index = pd.date_range(frame["timestamp"].iloc[0], frame["timestamp"].iloc[-1], freq="1s", tz="UTC")
    frame = frame.set_index("timestamp").reindex(index)
    frame["close"] = frame["close"].ffill()
    for column in ("open", "high", "low"):
        frame[column] = frame[column].fillna(frame["close"])
    for column in ("volume", "quote_volume", "trade_count", "taker_buy_volume"):
        frame[column] = frame[column].fillna(0.0)
    frame.index.name = "timestamp"
    frame = frame.reset_index()
    return frame.to_dict(orient="records")


class BinanceAggTradeFeed:
    """Reconnectable public WebSocket feed, aggregating trades to dense 1s bars."""

    def __init__(
        self,
        symbol: str = SYMBOL,
        seed_seconds: int = CONTEXT_SECONDS + MAX_FEATURE_LOOKBACK_SECONDS,
        seed_history: bool = True,
        buffer_seconds: int = 100_000,
    ) -> None:
        self.symbol = symbol.upper()
        self.seed_seconds = int(seed_seconds)
        self.seed_history = bool(seed_history)
        self._bars: deque[dict] = deque(maxlen=max(buffer_seconds, seed_seconds + 3000))
        self._seed: list[dict] = []
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._ws_thread: threading.Thread | None = None
        self._clock_thread: threading.Thread | None = None
        self._ws_app: websocket.WebSocketApp | None = None
        self._current: dict | None = None
        self._last_closed_second: int | None = None
        self._error: str | None = None
        self._started = False

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def status(self) -> str:
        if self._error:
            return f"Feed warning: {self._error}"
        return "Connected to Binance Spot WebSocket" if self._started else "Feed not started"

    @property
    def seed_bars(self) -> list[dict]:
        return [dict(row) for row in self._seed]

    def start(self) -> "BinanceAggTradeFeed":
        if self._started:
            return self
        if self.seed_history:
            try:
                self._seed = fetch_recent_klines(self.symbol, self.seed_seconds)
                with self._lock:
                    for row in self._seed:
                        self._bars.append(dict(row))
                    if self._seed:
                        self._last_closed_second = int(pd.Timestamp(self._seed[-1]["timestamp"]).timestamp())
            except Exception as exc:  # feed may still start without warm history
                self._error = f"Could not seed 1-second REST history: {exc}"
                LOG.warning(self._error)
        self._started = True
        self._ws_thread = threading.Thread(target=self._websocket_loop, name="binance-aggtrade", daemon=True)
        self._clock_thread = threading.Thread(target=self._clock_loop, name="binance-bar-clock", daemon=True)
        self._ws_thread.start()
        self._clock_thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        try:
            if self._ws_app is not None:
                self._ws_app.close()
        except Exception:
            pass

    def snapshot(self) -> list[dict]:
        with self._lock:
            rows = [dict(row) for row in self._bars]
            if self._current is not None:
                rows.append(self._bar_from_current(self._current))
        return rows

    def snapshot_frame(self) -> pd.DataFrame:
        rows = self.snapshot()
        if not rows:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "quote_volume", "trade_count", "taker_buy_volume"])
        frame = pd.DataFrame(rows).drop_duplicates("timestamp", keep="last").sort_values("timestamp")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        return frame.reset_index(drop=True)

    def bars_since(self, timestamp: pd.Timestamp | None) -> list[dict]:
        # Only finalized bars enter the model. The in-progress bar is shown in
        # snapshot_frame() but is never consumed and then silently revised.
        with self._lock:
            rows = [dict(row) for row in self._bars]
        if timestamp is None:
            return rows
        last = pd.Timestamp(timestamp)
        if last.tzinfo is None:
            last = last.tz_localize("UTC")
        else:
            last = last.tz_convert("UTC")
        return [row for row in rows if pd.Timestamp(row["timestamp"]) > last]

    @staticmethod
    def _bar_from_current(current: dict) -> dict:
        second = int(current["second"])
        return {
            "timestamp": pd.Timestamp(second, unit="s", tz="UTC"),
            "open": float(current["open"]),
            "high": float(current["high"]),
            "low": float(current["low"]),
            "close": float(current["close"]),
            "volume": float(current["volume"]),
            "quote_volume": float(current["quote_volume"]),
            "trade_count": int(current["trade_count"]),
            "taker_buy_volume": float(current["taker_buy_volume"]),
        }

    def _append_bar_locked(self, row: dict) -> None:
        ts = pd.Timestamp(row["timestamp"])
        sec = int(ts.timestamp())
        if self._last_closed_second is not None and sec <= self._last_closed_second:
            return
        self._bars.append(row)
        self._last_closed_second = sec

    def _append_flat_until_locked(self, inclusive_second: int) -> None:
        if not self._bars or self._last_closed_second is None:
            return
        previous_close = float(self._bars[-1]["close"])
        next_second = self._last_closed_second + 1
        while next_second <= inclusive_second:
            self._append_bar_locked(
                {
                    "timestamp": pd.Timestamp(next_second, unit="s", tz="UTC"),
                    "open": previous_close,
                    "high": previous_close,
                    "low": previous_close,
                    "close": previous_close,
                    "volume": 0.0,
                    "quote_volume": 0.0,
                    "trade_count": 0,
                    "taker_buy_volume": 0.0,
                }
            )
            next_second += 1

    def _finalize_current_locked(self) -> None:
        if self._current is None:
            return
        row = self._bar_from_current(self._current)
        self._append_bar_locked(row)
        self._current = None

    def _on_message(self, _ws, message: str) -> None:
        try:
            event = json.loads(message)
            second = int(event["T"]) // 1000
            price = float(event["p"])
            quantity = float(event["q"])
            buyer_is_maker = bool(event["m"])
        except Exception as exc:
            LOG.debug("Ignoring malformed Binance event: %s", exc)
            return
        if price <= 0.0 or quantity < 0.0:
            return
        with self._lock:
            if self._last_closed_second is not None and second <= self._last_closed_second:
                return
            if self._current is None:
                if self._last_closed_second is not None and second > self._last_closed_second + 1:
                    self._append_flat_until_locked(second - 1)
                self._current = {
                    "second": second,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": quantity,
                    "quote_volume": price * quantity,
                    "trade_count": 1,
                    "taker_buy_volume": 0.0 if buyer_is_maker else quantity,
                }
                return
            current_second = int(self._current["second"])
            if second < current_second:
                return
            if second > current_second:
                previous_close = float(self._current["close"])
                self._finalize_current_locked()
                self._append_flat_until_locked(second - 1)
                if self._last_closed_second is None:
                    self._last_closed_second = second - 1
                self._current = {
                    "second": second,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": quantity,
                    "quote_volume": price * quantity,
                    "trade_count": 1,
                    "taker_buy_volume": 0.0 if buyer_is_maker else quantity,
                }
                return
            self._current["high"] = max(float(self._current["high"]), price)
            self._current["low"] = min(float(self._current["low"]), price)
            self._current["close"] = price
            self._current["volume"] += quantity
            self._current["quote_volume"] += price * quantity
            self._current["trade_count"] += 1
            if not buyer_is_maker:
                self._current["taker_buy_volume"] += quantity

    def _on_error(self, _ws, error) -> None:
        self._error = str(error)
        LOG.warning("Binance WebSocket error: %s", error)

    def _on_close(self, _ws, status_code, message) -> None:
        LOG.info("Binance WebSocket closed: %s %s", status_code, message)

    def _websocket_loop(self) -> None:
        url = WS_URL.format(symbol_lower=self.symbol.lower())
        while not self._stop.is_set():
            app = websocket.WebSocketApp(url, on_message=self._on_message, on_error=self._on_error, on_close=self._on_close)
            self._ws_app = app
            try:
                app.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as exc:
                self._error = str(exc)
                LOG.warning("WebSocket reconnect after error: %s", exc)
            if not self._stop.is_set():
                self._stop.wait(2.0)

    def _clock_loop(self) -> None:
        while not self._stop.wait(0.25):
            current_second = int(time.time())
            with self._lock:
                if self._current is not None and current_second - int(self._current["second"]) >= 2:
                    self._finalize_current_locked()
                if self._last_closed_second is not None:
                    self._append_flat_until_locked(current_second - 1)


class LiveModelRunner:
    """Inference, forward-error measurement, and optional delayed online updates."""

    def __init__(
        self,
        checkpoint_path: str | Path | None,
        device: str = "auto",
        context_seconds: int = CONTEXT_SECONDS,
        horizon_seconds: int = HORIZON_SECONDS,
        online_update: bool = False,
        save_every_updates: int = 300,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        self.context_seconds = int(context_seconds)
        self.horizon_seconds = int(horizon_seconds)
        self.engine = RollingFeatureEngine(self.context_seconds, self.horizon_seconds)
        self.model = None
        self.scaler: FeatureScaler | None = None
        self.analyzer: MarketAnalyzer | None = None
        self.payload: dict | None = None
        self._optimizer = None
        self._loss_fn = torch.nn.HuberLoss(delta=1.0)
        self._online_enabled = bool(online_update)
        self._save_every_updates = max(1, int(save_every_updates))
        self._online_updates = 0
        self._bar_index = -1
        self._last_timestamp: pd.Timestamp | None = None
        self._pending: deque[dict] = deque()
        self.predictions: deque[dict] = deque(maxlen=100_000)
        self.evaluations: deque[dict] = deque(maxlen=100_000)
        self.latest: dict | None = None
        self.stabilizer = PredictionStabilizer()
        self.load_error: str | None = None
        if self.checkpoint_path and self.checkpoint_path.exists():
            try:
                self.model, self.scaler, self.analyzer, self.payload = load_checkpoint(self.checkpoint_path, self.device)
            except Exception as exc:
                self.load_error = str(exc)
                LOG.exception("Could not load checkpoint %s", self.checkpoint_path)
        if self.model is None:
            profile = choose_model_profile(detect_hardware())
            self.analyzer = MarketAnalyzer(profile.analyzer_windows)
            self._online_enabled = False
        elif self._online_enabled:
            self.set_online(True)

    @property
    def last_timestamp(self) -> pd.Timestamp | None:
        return self._last_timestamp

    @property
    def online_enabled(self) -> bool:
        return self._online_enabled

    def set_online(self, enabled: bool) -> None:
        self._online_enabled = bool(enabled and self.model is not None)
        if self._online_enabled and self._optimizer is None:
            self._optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5, weight_decay=1e-5)

    def warm_start(self, bars: Iterable[dict]) -> None:
        """Populate causal feature history without emitting replay forecasts."""
        for bar in bars:
            timestamp = pd.Timestamp(bar["timestamp"])
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize("UTC")
            else:
                timestamp = timestamp.tz_convert("UTC")
            if self._last_timestamp is not None and timestamp <= self._last_timestamp:
                continue
            self.engine.update({**bar, "timestamp": timestamp})
            self._bar_index += 1
            self._last_timestamp = timestamp

    def process_bar(self, bar: dict) -> dict | None:
        timestamp = pd.Timestamp(bar["timestamp"])
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            return None
        row = {**bar, "timestamp": timestamp}
        raw_feature = self.engine.update(row)
        self._bar_index += 1
        self._last_timestamp = timestamp
        if self.model is None or self.scaler is None or self.analyzer is None:
            return None
        # Need a 12-hour input whose oldest feature itself has a full 12-hour
        # causal lookback. The live dashboard seeds 24 hours for this reason.
        if self._bar_index < self.context_seconds + MAX_FEATURE_LOOKBACK_SECONDS - 1:
            return None
        history = self.engine.history_array(self.context_seconds)
        if history is None:
            return None
        normalized = self.scaler.transform(history)
        sequence = make_model_input(normalized, context_seconds=self.context_seconds)
        signals = self.analyzer.analyze(normalized)
        sequence_tensor = torch.from_numpy(sequence[None]).to(self.device)
        signal_tensor = torch.from_numpy(signals[None]).to(self.device)
        self.model.eval()
        with torch.inference_mode():
            predicted_pct = float(self.model(sequence_tensor, signal_tensor).reshape(-1)[0].cpu())
        raw_log_return = predicted_pct / TARGET_SCALE
        rv_index = FEATURE_NAMES.index("realized_vol_900s")
        realized_vol = max(float(raw_feature[rv_index]), 0.0) * math.sqrt(self.horizon_seconds / 900.0)
        stable_log_return = self.stabilizer.update(raw_log_return, realized_vol)
        price = float(row["close"])
        prediction = {
            "timestamp": timestamp,
            "current_price": price,
            "raw_forecast_price": price * math.exp(float(np.clip(raw_log_return, -0.5, 0.5))),
            "forecast_price": price * math.exp(float(np.clip(stable_log_return, -0.5, 0.5))),
            "raw_log_return": raw_log_return,
            "stable_log_return": stable_log_return,
            "forecast_timestamp": timestamp + pd.Timedelta(seconds=self.horizon_seconds),
            "matured_actual_price": None,
            "online_update": False,
        }
        self.predictions.append(prediction)
        self._pending.append(
            {
                "index": self._bar_index,
                "timestamp": timestamp,
                "price": price,
                "raw_forecast_price": prediction["raw_forecast_price"],
                "forecast_price": prediction["forecast_price"],
                "stable_log_return": stable_log_return,
            }
        )
        self.latest = prediction
        self._mature_labels(price, timestamp)
        return prediction

    def _mature_labels(self, actual_price: float, actual_timestamp: pd.Timestamp) -> None:
        while self._pending and self._bar_index - int(self._pending[0]["index"]) >= self.horizon_seconds:
            item = self._pending.popleft()
            error = float(item["forecast_price"] - actual_price)
            record = {
                "timestamp": item["timestamp"],
                "current_price": item["price"],
                "raw_forecast_price": item["raw_forecast_price"],
                "forecast_price": item["forecast_price"],
                "actual_timestamp": actual_timestamp,
                "actual_price": float(actual_price),
                "error": error,
                "absolute_error": abs(error),
            }
            self.evaluations.append(record)
            if self._online_enabled:
                self._online_train(item["price"], actual_price)
                record["online_update"] = True

    def _online_train(self, origin_price: float, future_price: float) -> None:
        if self.model is None or self.scaler is None or self.analyzer is None:
            return
        raw_context = self.engine.history_array(self.context_seconds, delay=self.horizon_seconds)
        if raw_context is None:
            return
        normalized = self.scaler.transform(raw_context)
        sequence = make_model_input(normalized, context_seconds=self.context_seconds)
        signals = self.analyzer.analyze(normalized)
        target_value = TARGET_SCALE * (math.log(future_price) - math.log(origin_price))
        x = torch.from_numpy(sequence[None]).to(self.device)
        signal = torch.from_numpy(signals[None]).to(self.device)
        target = torch.tensor([target_value], device=self.device, dtype=torch.float32)
        self.model.train()
        self._optimizer.zero_grad(set_to_none=True)
        predicted = self.model(x, signal)
        loss = self._loss_fn(predicted, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self._optimizer.step()
        self.model.eval()
        self._online_updates += 1
        if self._online_updates % self._save_every_updates == 0 and self.checkpoint_path and self.payload:
            destination = self.checkpoint_path.with_name(self.checkpoint_path.stem + "_live.pt")
            save_live_checkpoint(destination, self.model, self.payload)
            LOG.info("Saved delayed online checkpoint to %s", destination)

    def process_bars(self, bars: Iterable[dict]) -> list[dict]:
        produced = []
        for bar in bars:
            prediction = self.process_bar(bar)
            if prediction is not None:
                produced.append(prediction)
        return produced


def frame_from_bars(bars: Iterable[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(list(bars))
    if frame.empty:
        return frame
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
