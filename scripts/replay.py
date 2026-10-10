#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

def main() -> int:
    parser = argparse.ArgumentParser(description="Replay historical BTC bars and score 25-minute forecasts when labels mature")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/replay.csv"))
    parser.add_argument("--speed", type=float, default=0.0, help="Market seconds per wall-clock second; 0 runs as fast as possible")
    parser.add_argument("--max-seconds", type=int, default=None)
    parser.add_argument("--online-update", action="store_true", help="Fine-tune after each 25-minute delayed label matures")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    try:
        from bitcoin_predictor.simulator import run_replay
    except ImportError as exc:
        raise SystemExit("Replay dependencies are missing. Install the `[train,live]` extras and a compatible PyTorch build.") from exc
    result = run_replay(args.data_dir, args.checkpoint, args.output, args.speed, args.max_seconds, args.online_update)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
