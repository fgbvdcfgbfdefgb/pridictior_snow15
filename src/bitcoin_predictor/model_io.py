"""Model checkpoint loading/saving with architecture metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .features import FEATURE_DIM, FeatureScaler, MarketAnalyzer
from .hardware import ModelProfile
from .models import build_predictor


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[torch.nn.Module, FeatureScaler, MarketAnalyzer, dict[str, Any]]:
    checkpoint_path = Path(path)
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch versions before `weights_only` was introduced
        payload = torch.load(checkpoint_path, map_location=device)
    profile = ModelProfile.from_dict(payload["profile"])
    analyzer = MarketAnalyzer(profile.analyzer_windows)
    model = build_predictor(str(payload["kind"]), FEATURE_DIM, analyzer.output_dim, profile)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    scaler = FeatureScaler.from_dict(payload["feature_scaler"])
    return model, scaler, analyzer, payload


def save_live_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    payload: dict[str, Any],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    updated = dict(payload)
    updated["state_dict"] = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(updated, destination)
