"""Encoders for microscopy images and RNA vectors."""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

LOGGER = logging.getLogger(__name__)


class ImageEncoder(nn.Module):
    """ResNet18 backbone with a projection head."""

    def __init__(
        self,
        embedding_dim: int = 256,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        self.feature_dim = backbone.fc.in_features
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.head = nn.Linear(self.feature_dim, embedding_dim)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.backbone(pixel_values)
        features = features.flatten(1)
        embeddings = self.head(features)
        return F.normalize(embeddings, dim=-1)


class RNAMlpEncoder(nn.Module):
    """Simple MLP for dense RNA expression profiles."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        embedding_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, expression: torch.Tensor) -> torch.Tensor:
        embeddings = self.network(expression)
        return F.normalize(embeddings, dim=-1)


def create_image_encoder(embedding_dim: int, pretrained: bool = False) -> ImageEncoder:
    LOGGER.info(
        "Initialising image encoder (embed_dim=%s, pretrained=%s)",
        embedding_dim,
        pretrained,
    )
    return ImageEncoder(embedding_dim=embedding_dim, pretrained=pretrained)


def create_rna_encoder(
    input_dim: int,
    embedding_dim: int,
    hidden_dim: int = 512,
    dropout: float = 0.1,
) -> RNAMlpEncoder:
    LOGGER.info(
        "Initialising RNA encoder (input_dim=%s, embed_dim=%s)",
        input_dim,
        embedding_dim,
    )
    return RNAMlpEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        dropout=dropout,
    )


EncoderFactory = Callable[..., nn.Module]

__all__ = [
    "ImageEncoder",
    "RNAMlpEncoder",
    "create_image_encoder",
    "create_rna_encoder",
    "EncoderFactory",
]
