#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chronological one-pass BTC predictor training and candidate selection")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--sample-stride", type=int, default=60, help="Forecast-origin stride in seconds; use 1 for every second")
    parser.add_argument("--batch-size", type=int, default=None, help="Global batch size; auto-sized if omitted")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=None, help="Optional update cap per candidate (useful for smoke tests)")
    parser.add_argument("--calibration-days", type=int, default=7)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--validation-stride", type=int, default=None)
    parser.add_argument("--max-eval-examples", type=int, default=100000)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--candidates", nargs="+", choices=["mlp", "tcn", "gru"], default=["mlp", "tcn", "gru"])
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        from bitcoin_predictor.training import train_candidates
    except ImportError as exc:
        raise SystemExit("Training dependencies are missing. Install the `[train]` extra and a compatible PyTorch build.") from exc
    train_candidates(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        sample_stride=args.sample_stride,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        calibration_days=args.calibration_days,
        validation_fraction=args.validation_fraction,
        validation_stride=args.validation_stride,
        max_eval_examples=args.max_eval_examples,
        requested_device=args.device,
        candidates=args.candidates,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
