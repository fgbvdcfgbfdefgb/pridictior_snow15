#!/usr/bin/env python3
"""Offline Snowflake wrapper: reads only local files or an internal Snowflake stage."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def _get_from_internal_stage(stage: str, target: Path) -> None:
    try:
        from snowflake.snowpark.context import get_active_session

        session = get_active_session()
    except Exception as active_error:
        connection_name = os.getenv("SNOWFLAKE_CONNECTION_NAME")
        if not connection_name:
            raise RuntimeError(
                "Could not get an active Snowpark session. In a Snowflake notebook use its active session, "
                "or set SNOWFLAKE_CONNECTION_NAME for a configured connection."
            ) from active_error
        from snowflake.snowpark import Session

        session = Session.builder.configs({"connection_name": connection_name}).create()
    target.mkdir(parents=True, exist_ok=True)
    try:
        session.file.get(stage, str(target))
    finally:
        # If this script created its own connection, the caller can still close
        # it normally; active notebook sessions should not be closed here.
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline training wrapper for Snowflake local/staged Parquet data")
    parser.add_argument("--data-dir", type=Path, default=None, help="Local directory already containing daily Parquet chunks")
    parser.add_argument("--stage", default=None, help="Optional internal-stage path, for example @DB.SCHEMA.BTC_STAGE/BTCUSDT_1s/")
    parser.add_argument("--output-dir", type=Path, default=Path("models_snowflake"))
    parser.add_argument("--device", default="cpu", help="Defaults to CPU; use CUDA only in a GPU-enabled Snowflake runtime")
    args, passthrough = parser.parse_known_args()

    if bool(args.data_dir) == bool(args.stage):
        parser.error("Choose exactly one of --data-dir or --stage")
    if args.stage:
        staged_dir = Path(tempfile.mkdtemp(prefix="btc-stage-"))
        _get_from_internal_stage(args.stage, staged_dir)
        data_dir = staged_dir
    else:
        data_dir = args.data_dir
    if not data_dir.exists():
        parser.error(f"Data path does not exist: {data_dir}")

    command = [
        sys.executable,
        str(ROOT / "scripts" / "train.py"),
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(args.output_dir),
        "--device",
        args.device,
        *passthrough,
    ]
    # No calls to Binance, GitHub, or any external data endpoint are made here.
    subprocess.run(command, check=True, cwd=ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
