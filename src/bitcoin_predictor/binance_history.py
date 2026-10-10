"""Download Binance Vision aggTrades and write dense daily 1-second Parquet chunks."""

from __future__ import annotations

import argparse
import calendar
import csv
from datetime import date, datetime, time, timedelta, timezone
import json
import logging
import os
from pathlib import Path
import tempfile
import zipfile
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from .constants import SYMBOL

LOG = logging.getLogger("bitcoin_predictor.data")
CSV_BASE_NAMES = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "transact_time", "is_buyer_maker", "is_best_match"]
BAR_AGG = {
    "open": ("open", "first"),
    "high": ("high", "max"),
    "low": ("low", "min"),
    "close": ("close", "last"),
    "volume": ("quantity", "sum"),
    "quote_volume": ("quote_quantity", "sum"),
    "trade_count": ("price", "size"),
    "taker_buy_volume": ("taker_buy_quantity", "sum"),
}


class ArchiveNotFound(FileNotFoundError):
    pass


def parse_utc_date(value: str) -> date:
    return date.fromisoformat(value)


def _month_bounds(year: int, month: int) -> tuple[date, date]:
    last_day = calendar.monthrange(year, month)[1]
    return date(year, month, 1), date(year, month, last_day)


def _month_iter(start: date, end: date) -> Iterable[tuple[int, int]]:
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        yield year, month
        month += 1
        if month == 13:
            year += 1
            month = 1


def _download(url: str, timeout: tuple[int, int] = (15, 180)) -> Path:
    """Stream one archive to a temporary file; callers always remove it."""
    response = requests.get(url, stream=True, timeout=timeout, headers={"User-Agent": "btc-second-bars/0.1"})
    if response.status_code == 404:
        response.close()
        raise ArchiveNotFound(url)
    response.raise_for_status()
    temp = tempfile.NamedTemporaryFile(prefix="btc-aggtrades-", suffix=".zip", delete=False)
    try:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                temp.write(chunk)
    finally:
        temp.close()
        response.close()
    return Path(temp.name)


def _csv_member(archive: zipfile.ZipFile) -> str:
    candidates = [name for name in archive.namelist() if name.lower().endswith(".csv")]
    if not candidates:
        raise ValueError("Binance ZIP did not contain a CSV")
    return candidates[0]


def _aggregate_member(archive: zipfile.ZipFile, member: str, chunksize: int = 500_000) -> pd.DataFrame:
    with archive.open(member) as stream:
        first_line = stream.readline().decode("utf-8-sig", errors="replace").strip()
    if not first_line:
        return pd.DataFrame(columns=["timestamp", *BAR_AGG.keys()])
    header = not first_line.split(",", 1)[0].strip().isdigit()
    width = len(next(csv.reader([first_line])))
    names = CSV_BASE_NAMES[:width]
    if width < 6:
        raise ValueError(f"Unexpected aggTrades CSV with {width} columns")

    parts: list[pd.DataFrame] = []
    usecols = [1, 2, 5] + ([6] if width >= 7 else [])
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
            price = pd.to_numeric(chunk["price"], errors="coerce")
            quantity = pd.to_numeric(chunk["quantity"], errors="coerce").clip(lower=0.0)
            raw_time = pd.to_numeric(chunk["transact_time"], errors="coerce")
            valid = price.notna() & quantity.notna() & raw_time.notna() & (price > 0.0)
            if not valid.any():
                continue
            price = price.loc[valid].astype("float64")
            quantity = quantity.loc[valid].astype("float64")
            raw_time = raw_time.loc[valid].astype("int64")
            median_time = int(raw_time.iloc[len(raw_time) // 2])
            if median_time >= 10**17:
                unit = "ns"
            elif median_time >= 10**14:
                unit = "us"
            elif median_time >= 10**11:
                unit = "ms"
            else:
                unit = "s"
            timestamps = pd.to_datetime(raw_time, unit=unit, utc=True).dt.floor("s")

            if "is_buyer_maker" in chunk.columns:
                maker_raw = chunk.loc[valid, "is_buyer_maker"].astype(str).str.strip().str.lower()
                buyer_is_maker = maker_raw.isin({"true", "1", "1.0", "yes"}).to_numpy()
                taker_buy = np.where(~buyer_is_maker, quantity.to_numpy(), 0.0)
            else:
                # Older/alternate archives without the maker flag still yield OHLCV;
                # taker-buy volume is zero rather than fabricated.
                taker_buy = np.zeros(len(quantity), dtype=np.float64)

            trades = pd.DataFrame(
                {
                    "timestamp": timestamps.to_numpy(),
                    "price": price.to_numpy(),
                    "quantity": quantity.to_numpy(),
                    "quote_quantity": (price.to_numpy() * quantity.to_numpy()),
                    "taker_buy_quantity": taker_buy,
                }
            )
            grouped = (
                trades.groupby("timestamp", sort=True)
                .agg(
                    open=("price", "first"),
                    high=("price", "max"),
                    low=("price", "min"),
                    close=("price", "last"),
                    volume=("quantity", "sum"),
                    quote_volume=("quote_quantity", "sum"),
                    trade_count=("price", "size"),
                    taker_buy_volume=("taker_buy_quantity", "sum"),
                )
                .reset_index()
            )
            parts.append(grouped)

    if not parts:
        return pd.DataFrame(columns=["timestamp", *BAR_AGG.keys()])
    merged = pd.concat(parts, ignore_index=True)
    # A one-second bar can straddle CSV chunks; re-merge partial OHLC bars in
    # original file order (Binance archives are chronological).
    merged = (
        merged.groupby("timestamp", sort=True)
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
    return merged


def _aggregate_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        member = _csv_member(archive)
        return _aggregate_member(archive, member)


def _monthly_url(symbol: str, year: int, month: int) -> str:
    stamp = f"{year:04d}-{month:02d}"
    filename = f"{symbol}-aggTrades-{stamp}.zip"
    return f"https://data.binance.vision/data/spot/monthly/aggTrades/{symbol}/{filename}"


def _daily_url(symbol: str, day: date) -> str:
    stamp = day.isoformat()
    filename = f"{symbol}-aggTrades-{stamp}.zip"
    return f"https://data.binance.vision/data/spot/daily/aggTrades/{symbol}/{filename}"


def _load_existing_close(path: Path) -> float | None:
    try:
        frame = pd.read_parquet(path, columns=["close"])
        if frame.empty:
            return None
        return float(frame["close"].iloc[-1])
    except Exception:
        return None


def _daily_dense_bars(
    aggregate: pd.DataFrame,
    day: date,
    requested_start: date,
    requested_end: date,
    previous_close: float | None,
) -> tuple[pd.DataFrame | None, float | None]:
    utc = timezone.utc
    range_start = datetime.combine(max(day, requested_start), time(0, 0, 0), tzinfo=utc)
    range_end = datetime.combine(min(day, requested_end), time(23, 59, 59), tzinfo=utc)
    if range_end < range_start:
        return None, previous_close

    day_agg = aggregate.copy()
    if not day_agg.empty:
        day_agg["timestamp"] = pd.to_datetime(day_agg["timestamp"], utc=True)
        day_agg = day_agg.loc[
            (day_agg["timestamp"] >= pd.Timestamp(range_start))
            & (day_agg["timestamp"] <= pd.Timestamp(range_end))
        ].set_index("timestamp").sort_index()
    else:
        day_agg = pd.DataFrame(columns=list(BAR_AGG), index=pd.DatetimeIndex([], tz="UTC"))

    if day_agg.empty and previous_close is None:
        LOG.warning("No trades available for %s and no prior close; omitting leading empty bars", day)
        return None, previous_close

    if day_agg.empty:
        first_timestamp = pd.Timestamp(range_start)
    elif previous_close is None:
        # Do not invent a price before the first observed trade in the dataset.
        first_timestamp = max(pd.Timestamp(range_start), day_agg.index[0])
    else:
        first_timestamp = pd.Timestamp(range_start)
    seconds = pd.date_range(first_timestamp, pd.Timestamp(range_end), freq="1s", tz="UTC")
    dense = day_agg.reindex(seconds)

    close = dense["close"].astype("float64").ffill()
    if previous_close is not None:
        close = close.fillna(previous_close)
    if close.isna().all():
        return None, previous_close
    dense["close"] = close
    for column in ("open", "high", "low"):
        dense[column] = dense[column].fillna(dense["close"])
    for column in ("volume", "quote_volume", "trade_count", "taker_buy_volume"):
        dense[column] = dense[column].fillna(0.0)
    dense["trade_count"] = dense["trade_count"].astype("int32")
    dense.index.name = "timestamp"
    dense = dense.reset_index()
    end_close = float(dense["close"].iloc[-1]) if not dense.empty else previous_close
    return dense, end_close


def _raw_archive_root(output_dir: Path) -> Path:
    # Default output `data/processed/<dataset>` stores retained upstream ZIPs in
    # the sibling `data/raw/` directory; custom destinations use a sibling `raw/`.
    output_dir = Path(output_dir)
    if output_dir.parent.name == "processed":
        return output_dir.parent.parent / "raw"
    return output_dir.parent / "raw"


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
        raise ValueError(
            f"end must be a completed UTC date before {today_utc}; the archive may not contain today's full data. "
            "Use the live feed for current-day bars."
        )
    symbol = symbol.upper()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "source": "Binance Vision Spot aggTrades",
        "symbol": symbol,
        "bar_interval": "1 second UTC",
        "bar_semantics": "trades aggregated to OHLCV; empty seconds forward-filled with zero volume",
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
        month_folder = output_dir / f"{year:04d}-{month:02d}"
        month_folder.mkdir(parents=True, exist_ok=True)

        # If all selected days already exist, avoid redownloading the source.
        selected_days = [day_start + timedelta(days=i) for i in range((day_end - day_start).days + 1)]
        all_existing = all((month_folder / f"{d.day:02d}.parquet").exists() for d in selected_days)
        if all_existing and not overwrite:
            last_path = month_folder / f"{selected_days[-1].day:02d}.parquet"
            loaded_close = _load_existing_close(last_path)
            if loaded_close is not None:
                previous_close = loaded_close
            manifest["skipped_existing_files"].extend(
                str((month_folder / f"{current_day.day:02d}.parquet").relative_to(output_dir))
                for current_day in selected_days
            )
            LOG.info("Skipping existing month %04d-%02d", year, month)
            continue

        aggregate: pd.DataFrame | None = None
        monthly_complete = day_start == month_start and day_end == month_end
        url = _monthly_url(symbol, year, month)
        if monthly_complete:
            archive_path: Path | None = None
            try:
                archive_path = _download(url)
                aggregate = _aggregate_zip(archive_path)
                manifest["sources"].append(url)
                if keep_raw:
                    raw_dir = _raw_archive_root(output_dir) / "binance_aggTrades" / symbol
                    raw_dir.mkdir(parents=True, exist_ok=True)
                    raw_copy = raw_dir / f"{symbol}-aggTrades-{year:04d}-{month:02d}.zip"
                    os.replace(archive_path, raw_copy)
                    archive_path = None
            except ArchiveNotFound:
                LOG.info("Monthly archive unavailable; falling back to daily archives: %s", url)
            finally:
                if archive_path is not None:
                    archive_path.unlink(missing_ok=True)

        if aggregate is None:
            daily_parts: list[pd.DataFrame] = []
            missing_in_month: list[date] = []
            for current_day in selected_days:
                day_url = _daily_url(symbol, current_day)
                archive_path = None
                try:
                    archive_path = _download(day_url)
                    daily_parts.append(_aggregate_zip(archive_path))
                    manifest["sources"].append(day_url)
                    if keep_raw:
                        raw_dir = _raw_archive_root(output_dir) / "binance_aggTrades" / symbol
                        raw_dir.mkdir(parents=True, exist_ok=True)
                        raw_copy = raw_dir / f"{symbol}-aggTrades-{current_day.isoformat()}.zip"
                        os.replace(archive_path, raw_copy)
                        archive_path = None
                except ArchiveNotFound:
                    missing_in_month.append(current_day)
                    manifest["complete"] = False
                    manifest["missing_dates"].append(current_day.isoformat())
                    if not skip_missing:
                        raise ArchiveNotFound(f"No monthly or daily Binance archive for {current_day}: {day_url}")
                    LOG.warning("Skipping missing archive for %s", current_day)
                finally:
                    if archive_path is not None:
                        archive_path.unlink(missing_ok=True)
            aggregate = (
                pd.concat(daily_parts, ignore_index=True)
                if daily_parts
                else pd.DataFrame(columns=["timestamp", *BAR_AGG.keys()])
            )
            if not aggregate.empty:
                aggregate = (
                    aggregate.groupby("timestamp", sort=True)
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

        if not aggregate.empty:
            aggregate["timestamp"] = pd.to_datetime(aggregate["timestamp"], utc=True)
        for current_day in selected_days:
            target = month_folder / f"{current_day.day:02d}.parquet"
            if current_day.isoformat() in manifest["missing_dates"]:
                # Do not fabricate a whole missing day by carrying a stale close.
                previous_close = None
                continue
            if target.exists() and not overwrite:
                loaded_close = _load_existing_close(target)
                if loaded_close is not None:
                    previous_close = loaded_close
                manifest["skipped_existing_files"].append(str(target.relative_to(output_dir)))
                continue
            day_aggregate = aggregate
            if not aggregate.empty:
                mask = pd.to_datetime(aggregate["timestamp"], utc=True).dt.date == current_day
                day_aggregate = aggregate.loc[mask]
            dense, previous_close = _daily_dense_bars(day_aggregate, current_day, start, end, previous_close)
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
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download Binance BTC/USDT aggTrades and write 1-second Parquet bars")
    parser.add_argument("--start", required=True, help="Inclusive UTC date, YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="Inclusive UTC date, YYYY-MM-DD")
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--output-dir", default="data/processed/BTCUSDT_1s")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-missing", action="store_true", help="Record missing archives and continue (not recommended for final training)")
    parser.add_argument("--keep-raw", action="store_true", help="Keep upstream ZIP archives under data/raw (large; ignored by Git)")
    parser.add_argument("--max-months", type=int, default=None, help="Process at most this many calendar months for a staged download")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    manifest = download_history(
        start=parse_utc_date(args.start),
        end=parse_utc_date(args.end),
        output_dir=Path(args.output_dir),
        symbol=args.symbol,
        overwrite=args.overwrite,
        skip_missing=args.skip_missing,
        keep_raw=args.keep_raw,
        max_months=args.max_months,
    )
    LOG.info("Manifest: %s", Path(args.output_dir) / "manifest.json")
    LOG.info("Complete=%s output_days=%d missing_dates=%d", manifest["complete"], manifest["actual_output_days"], len(manifest["missing_dates"]))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
