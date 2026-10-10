"""Second-by-second chronological replay and matured forecast scoring."""

from __future__ import annotations

import csv
import math
from pathlib import Path
import time
from typing import Iterator

import pandas as pd

from .constants import TARGET_SCALE
from .realtime import LiveModelRunner
from .training import list_parquet_files, read_bars


def iter_bar_rows(data_dir: Path, max_seconds: int | None = None) -> Iterator[dict]:
    emitted = 0
    for path in list_parquet_files(data_dir):
        frame = read_bars(path)
        for row in frame.to_dict(orient="records"):
            yield row
            emitted += 1
            if max_seconds is not None and emitted >= max_seconds:
                return


def run_replay(
    data_dir: Path,
    checkpoint: Path,
    output_path: Path,
    speed: float = 0.0,
    max_seconds: int | None = None,
    online_update: bool = False,
) -> dict:
    if speed < 0:
        raise ValueError("speed must be >= 0; 0 means run as fast as possible")
    runner = LiveModelRunner(checkpoint, online_update=online_update)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    last_timestamp = None
    abs_price_error_sum = 0.0
    squared_price_error_sum = 0.0
    abs_return_error_sum = 0.0
    direction_hits = 0
    fields = [
        "prediction_timestamp",
        "target_timestamp",
        "current_price",
        "raw_forecast_price",
        "forecast_price_stabilized",
        "actual_price",
        "forecast_error_usd",
        "absolute_error_usd",
        "forecast_return_pct",
        "actual_return_pct",
    ]
    start = time.monotonic()
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for bar in iter_bar_rows(data_dir, max_seconds=max_seconds):
            timestamp = pd.Timestamp(bar["timestamp"])
            if last_timestamp is not None and speed > 0:
                delta = max(0.0, (timestamp - last_timestamp).total_seconds())
                delay = delta / speed
                if delay:
                    time.sleep(delay)
            runner.process_bar(bar)
            last_timestamp = timestamp
            # Labels are scored only when the matching 1,500-second future price arrives.
            while runner.evaluations:
                evaluation = runner.evaluations.popleft()
                current_price = float(evaluation["current_price"])
                actual_price = float(evaluation["actual_price"])
                forecast_price = float(evaluation["forecast_price"])
                raw_forecast_price = float(evaluation["raw_forecast_price"])
                price_error = forecast_price - actual_price
                forecast_return_pct = TARGET_SCALE * math.log(max(forecast_price, 1e-12) / max(current_price, 1e-12))
                actual_return_pct = TARGET_SCALE * math.log(max(actual_price, 1e-12) / max(current_price, 1e-12))
                writer.writerow(
                    {
                        "prediction_timestamp": evaluation["timestamp"].isoformat(),
                        "target_timestamp": evaluation["actual_timestamp"].isoformat(),
                        "current_price": current_price,
                        "raw_forecast_price": raw_forecast_price,
                        "forecast_price_stabilized": forecast_price,
                        "actual_price": actual_price,
                        "forecast_error_usd": price_error,
                        "absolute_error_usd": abs(price_error),
                        "forecast_return_pct": forecast_return_pct,
                        "actual_return_pct": actual_return_pct,
                    }
                )
                rows_written += 1
                abs_price_error_sum += abs(price_error)
                squared_price_error_sum += price_error * price_error
                abs_return_error_sum += abs(forecast_return_pct - actual_return_pct)
                direction_hits += int((forecast_return_pct >= 0) == (actual_return_pct >= 0))
                if rows_written % 10_000 == 0:
                    stream.flush()

    elapsed = time.monotonic() - start
    summary = {
        "checkpoint": str(checkpoint),
        "output": str(output_path),
        "matured_forecasts_written": rows_written,
        "seconds_elapsed": round(elapsed, 3),
        "online_updates": int(runner._online_updates),
        "model_load_error": runner.load_error,
        "note": "Forecasts are scored only when the 1,500-second forward label matures; no orders are placed.",
    }
    if rows_written:
        summary.update(
            {
                "mae_usd": abs_price_error_sum / rows_written,
                "rmse_usd": math.sqrt(squared_price_error_sum / rows_written),
                "mae_return_pct_points": abs_return_error_sum / rows_written,
                "directional_accuracy": direction_hits / rows_written,
            }
        )
    if runner.predictions:
        summary["last_prediction"] = {
            "timestamp": runner.predictions[-1]["timestamp"].isoformat(),
            "current_price": runner.predictions[-1]["current_price"],
            "forecast_price": runner.predictions[-1]["forecast_price"],
        }
    return summary
