"""GPU-ready price-predictor candidates; each predicts a 25-minute log return."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .hardware import ModelProfile


class PredictorBase(nn.Module):
    def __init__(self, feature_dim: int, analyzer_dim: int, hidden: int, dropout: float = 0.15) -> None:
        super().__init__()
        self.analyzer_encoder = nn.Sequential(
            nn.Linear(analyzer_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def combine(self, temporal: torch.Tensor, analyzer: torch.Tensor) -> torch.Tensor:
        signals = self.analyzer_encoder(analyzer)
        return self.head(torch.cat((temporal, signals), dim=-1)).squeeze(-1)


class MLPPricePredictor(PredictorBase):
    def __init__(self, feature_dim: int, analyzer_dim: int, profile: ModelProfile) -> None:
        super().__init__(feature_dim, analyzer_dim, profile.hidden)
        self.pool_size = 256
        self.temporal = nn.Sequential(
            nn.Linear(feature_dim * self.pool_size, profile.hidden),
            nn.GELU(),
            nn.LayerNorm(profile.hidden),
            nn.Dropout(0.2),
            nn.Linear(profile.hidden, profile.hidden),
            nn.GELU(),
        )

    def forward(self, sequence: torch.Tensor, analyzer: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool1d(sequence.transpose(1, 2), self.pool_size).flatten(1)
        return self.combine(self.temporal(pooled), analyzer)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.1) -> None:
        super().__init__()
        padding = dilation
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.block(x))


class TCNPricePredictor(PredictorBase):
    def __init__(self, feature_dim: int, analyzer_dim: int, profile: ModelProfile) -> None:
        super().__init__(feature_dim, analyzer_dim, profile.hidden)
        self.stem = nn.Conv1d(feature_dim, profile.channels, kernel_size=5, padding=2)
        self.blocks = nn.Sequential(
            *[
                ResidualTemporalBlock(profile.channels, dilation=2 ** (index % 6))
                for index in range(profile.depth)
            ]
        )
        self.projection = nn.Sequential(
            nn.Linear(profile.channels * 2, profile.hidden),
            nn.GELU(),
            nn.LayerNorm(profile.hidden),
        )

    def forward(self, sequence: torch.Tensor, analyzer: torch.Tensor) -> torch.Tensor:
        value = self.stem(sequence.transpose(1, 2))
        value = self.blocks(value)
        pooled = torch.cat((F.adaptive_avg_pool1d(value, 1).squeeze(-1), F.adaptive_max_pool1d(value, 1).squeeze(-1)), dim=-1)
        return self.combine(self.projection(pooled), analyzer)


class GRUPricePredictor(PredictorBase):
    def __init__(self, feature_dim: int, analyzer_dim: int, profile: ModelProfile) -> None:
        super().__init__(feature_dim, analyzer_dim, profile.hidden)
        self.hidden_size = profile.hidden
        self.pool_size = 512
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=self.hidden_size,
            num_layers=max(1, min(profile.depth, 3)),
            batch_first=True,
            dropout=0.1 if profile.depth > 1 else 0.0,
        )
        self.temporal_projection = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
        )

    def forward(self, sequence: torch.Tensor, analyzer: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool1d(sequence.transpose(1, 2), self.pool_size).transpose(1, 2)
        output, _ = self.gru(pooled)
        summary = torch.cat((output[:, -1], output.mean(dim=1)), dim=-1)
        return self.combine(self.temporal_projection(summary), analyzer)


def build_predictor(kind: str, feature_dim: int, analyzer_dim: int, profile: ModelProfile) -> nn.Module:
    candidates = {
        "mlp": MLPPricePredictor,
        "tcn": TCNPricePredictor,
        "gru": GRUPricePredictor,
    }
    try:
        constructor = candidates[kind.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown predictor {kind!r}; choose one of {sorted(candidates)}") from exc
    return constructor(feature_dim=feature_dim, analyzer_dim=analyzer_dim, profile=profile)
