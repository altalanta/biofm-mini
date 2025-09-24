"""Projection heads for aligning modality embeddings."""

from __future__ import annotations

from torch import nn


class TwoLayerProjector(nn.Module):
    """Small MLP used to map modality encoders into shared space."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features):  # type: ignore[override]
        return self.net(features)


__all__ = ["TwoLayerProjector"]
