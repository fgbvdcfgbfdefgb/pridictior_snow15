"""Download Binance Vision 1-second klines into dense daily Parquet chunks.

This is substantially smaller than re-aggregating every raw trade and preserves
native 1-second OHLCV, quote volume, trade count, and taker-buy volume.
"""
from __future__ import annotations

import argparse
import calendar
import csv
from datetime import date, datetime, time, timedelta, timezone
import json
import logging
import os
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from .constants import SYMBOL
from .binance_history import ArchiveNotFound, _daily_dense_bars, _download, _load_existing_close, _month_bounds, _month_iter, _raw_archive_root

LOG = logging.getLogger("bitcoin_predictor.klines")
KLINE_NAMES = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trade_count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]


def _month_url(symbol: str, year: int, month: int) -> str:
    stamp = f"{year:04d}-{month:02d}"
    return f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/1s/{symbol}-1s-{stamp}.zip"


def _day_url(symbol: str, day: date) -> str:
    return f"https://data.binance.vision/data/spot/daily/klines/{symbol}/1s/{symbol}-1s-{day.isoformat()}.zip"


def _unit_for_epoch(values: pd.Series) -> str:
    median = int(values.iloc[len(values) // 2])
    if median >= 10**17:
        return "ns"
    if median >= 10**14:
        return "us"
    if median >= 10**11:
        return "ms"
    return "s"


def _aggregate_kline_member(archive: zipfile.ZipFile, member: str, chunksize: int = 500_000) -> pd.DataFrame:
    with archive.open(member) as stream:
        first_line = stream.readline().decode("utf-8-sig", errors="replace").strip()
    if not first_line:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "quote_volume", "trade_count", "taker_buy_volume"])
    header = not first_line.split(",", 1)[0].strip().isdigit()
    width = len(next(csv.reader([first_line])))
    if width < 10:
        raise ValueError(f"Unexpected Binance 1-second kline CSV with {width} columns")
    names = KLINE_NAMES[:width]
    usecols = [0, 1, 2, 3, 4, 5, 7, 8, 9]
    parts: list[pd.DataFrame] = []
    with archive.open(member) as stream:
        reader = pd.read_csv(
            stream,
            header=None,
            names=names,
            usecols=usecols,
            skiprows=1 if header else 0,
            chunksize=chunksize,
        )
        for chunk in reader:
            if chunk.empty:
                continue
            open_time = pd.to_numeric(chunk["open_time"], errors="coerce")
            valid = open_time.notna()
            if not valid.any():
                continue
            epoch = open_time.loc[valid].astype("int64")
            unit = _unit_for_epoch(epoch)
            bars = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(epoch, unit=unit, utc=True).dt.floor("s").to_numpy(),
                    "open": pd.to_numeric(chunk.loc[valid, "open"], errors="coerce").to_numpy(),
                    "high": pd.to_numeric(chunk.loc[valid, "high"], errors="coerce").to_numpy(),
                    "low": pd.to_numeric(chunk.loc[valid, "low"], errors="coerce").to_numpy(),
                    "close": pd.to_numeric(chunk.loc[valid, "close"], errors="coerce").to_numpy(),
                    "volume": pd.to_numeric(chunk.loc[valid, "volume"], errors="coerce").fillna(0.0).to_numpy(),
                    "quote_volume": pd.to_numeric(chunk.loc[valid, "quote_volume"], errors="coerce").fillna(0.0).to_numpy(),
                    "trade_count": pd.to_numeric(chunk.loc[valid, "trade_count"], errors="coerce").fillna(0).to_numpy(),
                    "taker_buy_volume": pd.to_numeric(chunk.loc[valid, "taker_buy_volume"], errors="coerce").fillna(0.0).to_numpy(),
                }
            ).dropna(subset=["open", "high", "low", "close"])
            if not bars.empty:
                parts.append(bars)
    if not parts:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "quote_volume", "trade_count", "taker_buy_volume"])
    frame = pd.concat(parts, ignore_index=True).sort_values("timestamp")
    # Normally unique by second. Merge if an archive contains a duplicate row.
    frame = (
        frame.groupby("timestamp", sort=True)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            quote_volume=("quote_volume", "sum"),
            trade_count=("trade_count", "sum"),
            taker_buy_volume=("taker_buy_volume", "sum"),
        )
        .reset_index()
    )
    return frame


def _aggregate_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not members:
            raise ValueError(f"No CSV found in {path}")
        return _aggregate_kline_member(archive, members[0])


def download_history(
    start: date,
    end: date,
    output_dir: Path,
    symbol: str = SYMBOL,
    overwrite: bool = False,
    skip_missing: bool = False,
    keep_raw: bool = False,
    max_months: int | None = None,
) -> dict:
    if end < start:
        raise ValueError("end date must be on or after start date")
    today_utc = datetime.now(timezone.utc).date()
    if end >= today_utc:
        raise ValueError(f"end must be a completed UTC date before {today_utc}; use the live feed for current-day bars")
    symbol = symbol.upper()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": "Binance Vision Spot 1s klines",
        "symbol": symbol,
        "bar_interval": "1 second UTC",
        "bar_semantics": "native 1s OHLCV; missing seconds forward-filled at previous close with zero volume",
        "requested_start": start.isoformat(),
        "requested_end_inclusive": end.isoformat(),
        "complete": True,
        "missing_dates": [],
        "outputs": [],
        "skipped_existing_files": [],
        "sources": [],
    }
    previous_close: float | None = None
    processed_months = 0
    for year, month in _month_iter(start, end):
        if max_months is not None and processed_months >= max_months:
            manifest["complete"] = False
            manifest["stopped_early"] = True
            break
        processed_months += 1
        month_start, month_end = _month_bounds(year, month)
        day_start, day_end = max(start, month_start), min(end, month_end)
        month_dir = output_dir / f"{year:04d}-{month:02d}"
        month_dir.mkdir(parents=True, exist_ok=True)
        selected_days = [day_start + timedelta(days=i) for i in range((day_end - day_start).days + 1)]
        all_existing = all((month_dir / f"{day.day:02d}.parquet").exists() for day in selected_days)
        if all_existing and not overwrite:
            last_close = _load_existing_close(month_dir / f"{selected_days[-1].day:02d}.parquet")
            if last_close is not None:
                previous_close = last_close
            manifest["skipped_existing_files"].extend(
                str((month_dir / f"{day.day:02d}.parquet").relative_to(output_dir)) for day in selected_days
            )
            LOG.info("Skipping existing month %04d-%02d", year, month)
            continue

        aggregate: pd.DataFrame | None = None
        full_month = day_start == month_start and day_end == month_end
        monthly_url = _month_url(symbol, year, month)
        if full_month:
            archive_path: Path | None = None
            try:
                archive_path = _download(monthly_url)
                aggregate = _aggregate_zip(archive_path)
                manifest["sources"].append(monthly_url)
                if keep_raw:
                    raw_dir = _raw_archive_root(output_dir) / "binance_klines" / symbol
                    raw_dir.mkdir(parents=True, exist_ok=True)
                    raw_copy = raw_dir / f"{symbol}-1s-{year:04d}-{month:02d}.zip"
                    os.replace(archive_path, raw_copy)
                    archive_path = None
            except ArchiveNotFound:
                LOG.info("Monthly archive unavailable; falling back to daily files: %s", monthly_url)
            finally:
                if archive_path is not None:
                    archive_path.unlink(missing_ok=True)

        if aggregate is None:
            day_parts: list[pd.DataFrame] = []
            for day in selected_days:
                day_url = _day_url(symbol, day)
                archive_path = None
                try:
                    archive_path = _download(day_url)
                    day_parts.append(_aggregate_zip(archive_path))
                    manifest["sources"].append(day_url)
                    if keep_raw:
                        raw_dir = _raw_archive_root(output_dir) / "binance_klines" / symbol
                        raw_dir.mkdir(parents=True, exist_ok=True)
                        raw_copy = raw_dir / f"{symbol}-1s-{day.isoformat()}.zip"
                        os.replace(archive_path, raw_copy)
                        archive_path = None
                except ArchiveNotFound:
                    manifest["complete"] = False
                    manifest["missing_dates"].append(day.isoformat())
                    if not skip_missing:
                        raise ArchiveNotFound(f"No monthly or daily 1s kline archive for {day}: {day_url}")
                    LOG.warning("Skipping missing source archive for %s", day)
                finally:
                    if archive_path is not None:
                        archive_path.unlink(missing_ok=True)
            aggregate = pd.concat(day_parts, ignore_index=True) if day_parts else pd.DataFrame()
            if not aggregate.empty:
                aggregate = (
                    aggregate.groupby("timestamp", sort=True)
                    .agg(
                        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
                        volume=("volume", "sum"), quote_volume=("quote_volume", "sum"),
                        trade_count=("trade_count", "sum"), taker_buy_volume=("taker_buy_volume", "sum"),
                    )
                    .reset_index()
                )

        if not aggregate.empty:
            aggregate["timestamp"] = pd.to_datetime(aggregate["timestamp"], utc=True)
        for day in selected_days:
            target = month_dir / f"{day.day:02d}.parquet"
            if day.isoformat() in manifest["missing_dates"]:
                previous_close = None
                continue
            if target.exists() and not overwrite:
                last_close = _load_existing_close(target)
                if last_close is not None:
                    previous_close = last_close
                manifest["skipped_existing_files"].append(str(target.relative_to(output_dir)))
                continue
            day_aggregate = aggregate
            if not aggregate.empty:
                day_aggregate = aggregate.loc[pd.to_datetime(aggregate["timestamp"], utc=True).dt.date == day]
            dense, previous_close = _daily_dense_bars(day_aggregate, day, start, end, previous_close)
            if dense is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            dense.to_parquet(target, index=False, compression="zstd")
            manifest["outputs"].append(
                {
                    "file": str(target.relative_to(output_dir)),
                    "rows": int(len(dense)),
                    "first_timestamp": str(dense["timestamp"].iloc[0]),
                    "last_timestamp": str(dense["timestamp"].iloc[-1]),
                    "bytes": int(target.stat().st_size),
                }
            )
            LOG.info("Wrote %s (%s rows)", target, f"{len(dense):,}")

    manifest["generated_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["actual_output_days"] = len(set(manifest["skipped_existing_files"]) | {item["file"] for item in manifest["outputs"]})
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download Binance BTC/USDT 1-second kline archives to daily Parquet")
    parser.add_argument("--start", required=True, help="Inclusive UTC date, YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="Inclusive completed UTC date, YYYY-MM-DD")
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--output-dir", default="data/processed/BTCUSDT_1s")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--max-months", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    manifest = download_history(
        start=date.fromisoformat(args.start), end=date.fromisoformat(args.end), output_dir=Path(args.output_dir),
        symbol=args.symbol, overwrite=args.overwrite, skip_missing=args.skip_missing,
        keep_raw=args.keep_raw, max_months=args.max_months,
    )
    LOG.info("Manifest: %s", Path(args.output_dir) / "manifest.json")
    LOG.info("Complete=%s output_days=%d missing_dates=%d", manifest["complete"], manifest["actual_output_days"], len(manifest["missing_dates"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
