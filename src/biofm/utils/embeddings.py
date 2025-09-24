"""Helper functions for embedding export and loading."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from biofm.training.utils import create_paired_dataloader, get_device
from biofm.utils.pipeline import build_model, ensure_data, load_bundle
from biofm.utils.seeds import seed_everything

LOGGER = logging.getLogger(__name__)


def _load_checkpoint(model: torch.nn.Module, checkpoint: Path) -> None:
    if not checkpoint.exists():
        LOGGER.warning("Checkpoint %s not found; skipping load", checkpoint)
        return
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["model_state"])
    LOGGER.info("Loaded checkpoint %s", checkpoint)


def export_embeddings(cfg: DictConfig, save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    project_root = Path(cfg.paths.project_root)
    seed_everything(int(cfg.project.seed))
    ensure_data(cfg, project_root)
    bundle = load_bundle(cfg)

    dataloader = create_paired_dataloader(
        bundle=bundle,
        batch_size=int(cfg.eval.batch_size),
        image_size=int(cfg.data.image_size),
        augment=False,
        select_hvg=int(cfg.data.select_hvg) if cfg.data.select_hvg else None,
    )
    rna_input_dim = dataloader.dataset.rna_dataset.matrix.shape[1]  # type: ignore[attr-defined]
    model = build_model(cfg, rna_input_dim)
    _load_checkpoint(model, Path(cfg.eval.checkpoint))
    device = get_device()
    model.to(device)
    model.eval()

    image_embeddings: list[np.ndarray] = []
    rna_embeddings: list[np.ndarray] = []
    sample_ids: list[str] = []

    with torch.no_grad():
        for batch in dataloader:
            pixels = batch["pixel_values"].to(device)
            expression = batch["expression"].to(device)
            image_repr = model.image_projector(model.encode_image(pixels))
            rna_repr = model.rna_projector(model.encode_rna(expression))
            image_embeddings.append(
                torch.nn.functional.normalize(image_repr, dim=-1).cpu().numpy()
            )
            rna_embeddings.append(
                torch.nn.functional.normalize(rna_repr, dim=-1).cpu().numpy()
            )
            sample_ids.extend(batch["sample_id"])

    image_df = pd.DataFrame(np.vstack(image_embeddings), index=sample_ids)
    rna_df = pd.DataFrame(np.vstack(rna_embeddings), index=sample_ids)

    if save:
        processed_dir = Path(cfg.paths.data_dir) / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        image_df.to_parquet(processed_dir / "image_embeddings.parquet")
        rna_df.to_parquet(processed_dir / "rna_embeddings.parquet")
        LOGGER.info("Embeddings saved under %s", processed_dir)

    return image_df, rna_df


def load_embeddings_from_disk(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    image_path = data_dir / "processed" / "image_embeddings.parquet"
    rna_path = data_dir / "processed" / "rna_embeddings.parquet"
    if not image_path.exists() or not rna_path.exists():
        raise FileNotFoundError("Embeddings not found. Run embed.py or export_embeddings first.")
    return pd.read_parquet(image_path), pd.read_parquet(rna_path)


__all__ = ["export_embeddings", "load_embeddings_from_disk"]
