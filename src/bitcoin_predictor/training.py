"""One-pass chronological warm-start training and candidate selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
import math
import os
from pathlib import Path
import shutil
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from .constants import CONTEXT_SECONDS, HORIZON_SECONDS, TARGET_SCALE
from .features import FEATURE_DIM, FEATURE_NAMES, RETURN_LAGS, FeatureScaler, MarketAnalyzer, build_features, make_model_input
from .hardware import ModelProfile, choose_model_profile, detect_hardware
from .models import build_predictor

LOG = logging.getLogger("bitcoin_predictor.training")
BAR_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trade_count",
    "taker_buy_volume",
]


@dataclass
class StreamExample:
    timestamp: pd.Timestamp
    sequence: np.ndarray
    analyzer: np.ndarray
    target_pct: float
    current_price: float
    future_price: float


def list_parquet_files(data_dir: Path) -> list[Path]:
    files = sorted(Path(data_dir).rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No .parquet files found under {data_dir}")
    return files


def read_bars(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "timestamp" not in frame.columns:
        if isinstance(frame.index, pd.DatetimeIndex):
            frame = frame.reset_index(names="timestamp")
        else:
            raise ValueError(f"{path} has no timestamp column")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    defaults = {"quote_volume": 0.0, "trade_count": 0, "taker_buy_volume": 0.0}
    for column, default in defaults.items():
        if column not in frame:
            frame[column] = default
    required = ["open", "high", "low", "close", "volume", "quote_volume", "trade_count", "taker_buy_volume"]
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing bar columns: {sorted(missing)}")
    frame = frame[["timestamp", *required]].reset_index(drop=True)
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close", "volume"])
    frame = frame.loc[(frame["close"] > 0) & (frame["open"] > 0)].reset_index(drop=True)
    if len(frame) > 1:
        gaps = frame["timestamp"].diff().iloc[1:] != pd.Timedelta(seconds=1)
        if bool(gaps.any()):
            raise ValueError(
                f"{path} is not dense 1-second UTC data ({int(gaps.sum())} gaps or duplicate intervals). "
                "Regenerate with the provided downloader before training/replay."
            )
    return frame


def _join_contiguous(history: pd.DataFrame, current: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Join a rolling context to the next day, resetting over unrepresented gaps."""
    if history.empty:
        return current.reset_index(drop=True), 0
    if current.empty:
        return history.reset_index(drop=True), len(history)
    last_ts = pd.Timestamp(history["timestamp"].iloc[-1])
    first_ts = pd.Timestamp(current["timestamp"].iloc[0])
    if first_ts - last_ts != pd.Timedelta(seconds=1):
        history = history.iloc[0:0]
    old_count = len(history)
    joined = pd.concat((history, current), ignore_index=True)
    return joined, old_count


def fit_feature_scaler(files: Sequence[Path], calibration_days: int = 7) -> FeatureScaler:
    """Fit standardization on only the earliest training files; never validation."""
    history = pd.DataFrame(columns=BAR_COLUMNS)
    samples: list[np.ndarray] = []
    chosen = list(files[: max(1, int(calibration_days))])
    for path in chosen:
        current = read_bars(path)
        segment, old_count = _join_contiguous(history, current)
        features = build_features(segment)
        block_start = max(old_count, max(RETURN_LAGS))
        block = features[block_start::30]
        if len(block):
            samples.append(block)
        history = segment.iloc[-(max(RETURN_LAGS) + 1) :].copy().reset_index(drop=True)
    if not samples:
        raise ValueError("Could not obtain feature rows to fit the training scaler")
    return FeatureScaler.fit(np.concatenate(samples, axis=0))


def iter_stream_examples(
    files: Sequence[Path],
    scaler: FeatureScaler,
    analyzer: MarketAnalyzer,
    sample_stride: int = 60,
    context_seconds: int = CONTEXT_SECONDS,
    horizon_seconds: int = HORIZON_SECONDS,
    start_at: pd.Timestamp | None = None,
    max_examples: int | None = None,
) -> Iterator[StreamExample]:
    """Yield once-only supervised examples as labels mature in market time.

    For a sample originating at time t, it is yielded only while processing the
    bar at t+horizon. Context is causal and label delay is exactly the configured
    horizon. A stride > 1 is an offline compute shortcut; set it to 1 for every
    second. Predictions in the replay/live runner are not stride-limited.
    """
    if sample_stride < 1:
        raise ValueError("sample_stride must be at least one second")
    history = pd.DataFrame(columns=BAR_COLUMNS)
    emitted = 0
    for path in files:
        current = read_bars(path)
        segment, new_start = _join_contiguous(history, current)
        if len(segment) <= new_start:
            continue
        raw_features = build_features(segment)
        normalized = scaler.transform(raw_features)
        timestamps = pd.to_datetime(segment["timestamp"], utc=True).reset_index(drop=True)
        closes = segment["close"].to_numpy(dtype=np.float64)

        # `current_index` is the market time at which the earlier target is known.
        for current_index in range(new_start, len(segment)):
            origin = current_index - horizon_seconds
            # Features at the oldest input second need their own full maximum
            # lookback; otherwise the coarse history would contain cold-start rows.
            if origin - context_seconds + 1 < max(RETURN_LAGS):
                continue
            origin_timestamp = pd.Timestamp(timestamps.iloc[origin])
            if start_at is not None and origin_timestamp < start_at:
                continue
            if int(origin_timestamp.value // 1_000_000_000) % sample_stride != 0:
                continue
            context = normalized[origin - context_seconds + 1 : origin + 1]
            if len(context) != context_seconds:
                continue
            future_price = float(closes[current_index])
            current_price = float(closes[origin])
            target = TARGET_SCALE * (math.log(future_price) - math.log(current_price))
            if not math.isfinite(target):
                continue
            yield StreamExample(
                timestamp=origin_timestamp,
                sequence=make_model_input(context, context_seconds=context_seconds),
                analyzer=analyzer.analyze(context),
                target_pct=float(target),
                current_price=current_price,
                future_price=future_price,
            )
            emitted += 1
            if max_examples is not None and emitted >= max_examples:
                return
        history = segment.iloc[-(context_seconds + horizon_seconds + max(RETURN_LAGS) + 1) :].copy().reset_index(drop=True)


def _batch_tensor(examples: Sequence[StreamExample], device: torch.device) -> tuple[torch.Tensor, ...]:
    x = torch.from_numpy(np.stack([example.sequence for example in examples])).to(device=device, dtype=torch.float32)
    analyzer = torch.from_numpy(np.stack([example.analyzer for example in examples])).to(device=device, dtype=torch.float32)
    target = torch.tensor([example.target_pct for example in examples], device=device, dtype=torch.float32)
    return x, analyzer, target


def _forward(model: nn.Module, sequence: torch.Tensor, signals: torch.Tensor) -> torch.Tensor:
    return model(sequence, signals)


def _train_one_candidate(
    kind: str,
    train_files: Sequence[Path],
    scaler: FeatureScaler,
    profile: ModelProfile,
    analyzer: MarketAnalyzer,
    device: torch.device,
    rank: int,
    world_size: int,
    requested_batch_size: int,
    sample_stride: int,
    learning_rate: float,
    weight_decay: float,
    max_steps: int | None,
) -> tuple[nn.Module, int, float]:
    torch.manual_seed(1701 + (abs(hash(kind)) % 10000))
    base_model = build_predictor(kind, FEATURE_DIM, analyzer.output_dim, profile).to(device)
    wrapped: nn.Module = base_model
    use_ddp = world_size > 1
    if use_ddp:
        wrapped = DistributedDataParallel(
            base_model,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
            find_unused_parameters=False,
        )
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.HuberLoss(delta=1.0)
    local_batch = max(1, int(requested_batch_size) // world_size)
    global_batch_size = local_batch * world_size
    batch: list[StreamExample] = []
    steps = 0
    losses: list[float] = []
    wrapped.train()

    stream = iter_stream_examples(train_files, scaler, analyzer, sample_stride=sample_stride)
    for example in stream:
        batch.append(example)
        if len(batch) < global_batch_size:
            continue
        local = batch[rank * local_batch : (rank + 1) * local_batch]
        batch = []
        x, signals, target = _batch_tensor(local, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = _forward(wrapped, x, signals)
        loss = loss_fn(prediction, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=1.0)
        optimizer.step()
        steps += 1
        losses.append(float(loss.detach().cpu()))
        if rank == 0 and steps % 100 == 0:
            LOG.info("candidate=%s update=%d loss=%.5f", kind, steps, losses[-1])
        if max_steps is not None and steps >= max_steps:
            break

    if steps == 0:
        raise ValueError(
            "No full training batch was produced. Add more daily data, lower --sample-stride/--batch-size, "
            "or reduce the context/horizon only for a controlled test."
        )
    base_model.eval()
    return base_model, steps, float(np.mean(losses))


def evaluate_model(
    model: nn.Module,
    files: Sequence[Path],
    scaler: FeatureScaler,
    analyzer: MarketAnalyzer,
    device: torch.device,
    sample_stride: int,
    batch_size: int = 32,
    max_examples: int | None = 100_000,
) -> dict[str, float | int]:
    model.eval()
    examples: list[StreamExample] = []
    predictions: list[float] = []
    actuals: list[float] = []
    generator = iter_stream_examples(
        files,
        scaler,
        analyzer,
        sample_stride=sample_stride,
        max_examples=max_examples,
    )
    with torch.no_grad():
        for example in generator:
            examples.append(example)
            if len(examples) < batch_size:
                continue
            x, signals, target = _batch_tensor(examples, device)
            predictions.extend(_forward(model, x, signals).detach().cpu().numpy().astype(float).tolist())
            actuals.extend(target.detach().cpu().numpy().astype(float).tolist())
            examples = []
        if examples:
            x, signals, target = _batch_tensor(examples, device)
            predictions.extend(_forward(model, x, signals).detach().cpu().numpy().astype(float).tolist())
            actuals.extend(target.detach().cpu().numpy().astype(float).tolist())
    if not predictions:
        raise ValueError("Validation stream produced no examples; use a longer validation period")
    pred = np.asarray(predictions, dtype=np.float64)
    actual = np.asarray(actuals, dtype=np.float64)
    mae = float(np.mean(np.abs(pred - actual)))
    rmse = float(np.sqrt(np.mean(np.square(pred - actual))))
    direction = float(np.mean(np.sign(pred) == np.sign(actual)))
    instability = float(np.mean(np.abs(np.diff(pred)))) if len(pred) > 1 else 0.0
    score = mae + 0.25 * rmse + 0.10 * instability
    return {
        "examples": int(len(pred)),
        "mae_pct_points": mae,
        "rmse_pct_points": rmse,
        "directional_accuracy": direction,
        "mean_abs_prediction_change_pct_points": instability,
        "selection_score": float(score),
    }


def _checkpoint_payload(
    model: nn.Module,
    kind: str,
    profile: ModelProfile,
    scaler: FeatureScaler,
    metrics: dict,
    train_steps: int,
    sample_stride: int,
    training_end: str,
) -> dict:
    return {
        "state_dict": model.state_dict(),
        "kind": kind,
        "profile": profile.to_dict(),
        "feature_names": list(FEATURE_NAMES),
        "feature_scaler": scaler.to_dict(),
        "analyzer_windows": list(profile.analyzer_windows),
        "context_seconds": CONTEXT_SECONDS,
        "horizon_seconds": HORIZON_SECONDS,
        "target_scale": TARGET_SCALE,
        "sample_stride_seconds": int(sample_stride),
        "train_updates": int(train_steps),
        "training_end": training_end,
        "validation_metrics": metrics,
    }


def train_candidates(
    data_dir: Path,
    output_dir: Path,
    sample_stride: int = 60,
    batch_size: int | None = None,
    learning_rate: float = 2e-4,
    weight_decay: float = 1e-4,
    max_steps: int | None = None,
    calibration_days: int = 7,
    validation_fraction: float = 0.15,
    validation_stride: int | None = None,
    max_eval_examples: int | None = 100_000,
    requested_device: str = "auto",
    candidates: Sequence[str] = ("mlp", "tcn", "gru"),
) -> dict:
    """Train/evaluate all candidates in chronological one-pass DDP mode."""
    import torch.distributed as dist

    hardware = detect_hardware()  # intentionally before model/profile construction
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    cuda_ok = bool(torch.cuda.is_available())
    if requested_device == "auto":
        device = torch.device(f"cuda:{local_rank}" if cuda_ok else "cpu")
    else:
        device = torch.device(requested_device)
    if device.type == "cuda":
        if not cuda_ok:
            raise RuntimeError("CUDA was requested but this process has no visible CUDA device")
        torch.cuda.set_device(device)
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

    profile = choose_model_profile(hardware)
    effective_batch = int(batch_size or profile.batch_size)
    output_dir = Path(output_dir)
    candidate_dir = output_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    files = list_parquet_files(Path(data_dir))
    if len(files) < 2:
        raise ValueError("Need at least two daily Parquet chunks for a chronological train/validation split")
    split = max(1, min(len(files) - 1, int(len(files) * (1.0 - validation_fraction))))
    train_files, validation_files = files[:split], files[split:]
    scaler = fit_feature_scaler(train_files, calibration_days=calibration_days)
    analyzer = MarketAnalyzer(profile.analyzer_windows)
    if rank == 0:
        print(
            json.dumps(
                {
                    "hardware": hardware,
                    "selected_profile": profile.to_dict(),
                    "device": str(device),
                    "world_size": world_size,
                    "train_days": len(train_files),
                    "validation_days": len(validation_files),
                    "train_range": [train_files[0].as_posix(), train_files[-1].as_posix()],
                    "validation_range": [validation_files[0].as_posix(), validation_files[-1].as_posix()],
                },
                indent=2,
            ),
            flush=True,
        )

    results: list[dict] = []
    try:
        for kind in candidates:
            model, steps, mean_loss = _train_one_candidate(
                kind=kind,
                train_files=train_files,
                scaler=scaler,
                profile=profile,
                analyzer=analyzer,
                device=device,
                rank=rank,
                world_size=world_size,
                requested_batch_size=effective_batch,
                sample_stride=sample_stride,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                max_steps=max_steps,
            )
            if world_size > 1:
                dist.barrier()
            if rank == 0:
                metrics = evaluate_model(
                    model,
                    validation_files,
                    scaler,
                    analyzer,
                    device,
                    sample_stride=validation_stride or sample_stride,
                    max_examples=max_eval_examples,
                )
                metrics["mean_training_huber_loss"] = mean_loss
                metrics["train_updates"] = steps
                # The validation forecast origin range is chronological and held out.
                training_end = str(read_bars(train_files[-1])["timestamp"].iloc[-1])
                payload = _checkpoint_payload(model, kind, profile, scaler, metrics, steps, sample_stride, training_end)
                checkpoint = candidate_dir / f"{kind}.pt"
                torch.save(payload, checkpoint)
                metrics_record = {"kind": kind, "checkpoint": str(checkpoint), **metrics}
                (candidate_dir / f"{kind}.json").write_text(json.dumps(metrics_record, indent=2) + "\n", encoding="utf-8")
                results.append(metrics_record)
                LOG.info("candidate=%s validation=%s", kind, json.dumps(metrics, sort_keys=True))
            if world_size > 1:
                dist.barrier()

        if rank == 0:
            if not results:
                raise RuntimeError("No candidate results were produced on rank zero")
            best = min(results, key=lambda record: record["selection_score"])
            shutil.copy2(candidate_dir / f"{best['kind']}.pt", output_dir / "best.pt")
            summary = {
                "best_kind": best["kind"],
                "best_checkpoint": "best.pt",
                "selection_rule": "min(MAE + 0.25*RMSE + 0.10*mean_abs_prediction_change), all in percentage points",
                "candidate_results": results,
                "profile": profile.to_dict(),
                "hardware": hardware,
                "horizon_seconds": HORIZON_SECONDS,
                "context_seconds": CONTEXT_SECONDS,
                "sample_stride_seconds": int(sample_stride),
            }
            (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            LOG.info("Selected %s; checkpoint saved to %s", best["kind"], output_dir / "best.pt")
            return summary
        return {}
    finally:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()
