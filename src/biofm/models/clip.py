"""CLIP-style contrastive training for multimodal embeddings."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from biofm.models.encoders import ImageEncoder, RNAMlpEncoder
from biofm.models.projector import TwoLayerProjector


class BioFMClipModel(nn.Module):
    """Contrastive multimodal model pairing microscopy and RNA encoders."""

    def __init__(
        self,
        image_encoder: ImageEncoder,
        rna_encoder: RNAMlpEncoder,
        projector_dim: int = 256,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.rna_encoder = rna_encoder

        # Projectors map from encoder embedding_dim to shared projector_dim
        image_embedding_dim = image_encoder.head.out_features
        rna_embedding_dim = rna_encoder.network[-1].out_features

        self.image_projector = TwoLayerProjector(
            input_dim=image_embedding_dim,
            hidden_dim=projector_dim,
            output_dim=projector_dim,
        )
        self.rna_projector = TwoLayerProjector(
            input_dim=rna_embedding_dim,
            hidden_dim=projector_dim,
            output_dim=projector_dim,
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / temperature)))

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(pixel_values)

    def encode_rna(self, expression: torch.Tensor) -> torch.Tensor:
        return self.rna_encoder(expression)

    def forward(
        self, pixel_values: torch.Tensor, expression: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        image_embed = self.encode_image(pixel_values)
        rna_embed = self.encode_rna(expression)
        image_proj = F.normalize(self.image_projector(image_embed), dim=-1)
        rna_proj = F.normalize(self.rna_projector(rna_embed), dim=-1)
        logits = torch.matmul(image_proj, rna_proj.t()) * self.logit_scale.exp()
        targets = torch.arange(logits.size(0), device=logits.device)
        loss_i = F.cross_entropy(logits, targets)
        loss_t = F.cross_entropy(logits.t(), targets)
        loss = (loss_i + loss_t) / 2
        return {
            "loss": loss,
            "image_embeddings": image_proj.detach(),
            "rna_embeddings": rna_proj.detach(),
            "logits": logits.detach(),
        }


__all__ = ["BioFMClipModel"]
