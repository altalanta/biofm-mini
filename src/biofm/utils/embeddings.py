"""Helper functions for embedding export and loading."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from biofm.configuration import BioFMConfig
from biofm.deps import require_torch
from biofm.training.utils import create_paired_dataloader
from biofm.utils.pipeline import build_model, load_bundle

LOGGER = logging.getLogger(__name__)


def _load_checkpoint(model: object, checkpoint: Path | None) -> None:
    torch_module = require_torch()
    if checkpoint is None:
        LOGGER.info("No checkpoint provided; using randomly initialised weights")
        return
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint {checkpoint} not found")
    state = torch_module.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["model_state"])
    LOGGER.info("Loaded checkpoint %s", checkpoint)


def export_embeddings(
    config: BioFMConfig,
    device: object,
    checkpoint: Path | None = None,
    batch_size: int | None = None,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate and optionally persist modality-specific embeddings."""

    bundle = load_bundle(config)
    eval_batch_size = batch_size or int(config.eval.batch_size)
    dataloader = create_paired_dataloader(
        bundle=bundle,
        batch_size=eval_batch_size,
        image_size=int(config.data.image_size),
        augment=False,
        select_hvg=int(config.data.select_hvg) if config.data.select_hvg else None,
        num_workers=int(config.data.num_workers),
        pin_memory=device.type == "cuda",
    )
    rna_input_dim = dataloader.dataset.rna_dataset.matrix.shape[1]  # type: ignore[attr-defined]
    model = build_model(config, rna_input_dim)
    _load_checkpoint(model, checkpoint)
    torch_module = require_torch()
    model.to(device)
    model.eval()

    image_embeddings: list[np.ndarray] = []
    rna_embeddings: list[np.ndarray] = []
    sample_ids: list[str] = []

    with torch_module.no_grad():
        for batch in dataloader:
            pixels = batch["pixel_values"].to(device)
            expression = batch["expression"].to(device)
            image_repr = model.image_projector(model.encode_image(pixels))
            rna_repr = model.rna_projector(model.encode_rna(expression))
            image_embeddings.append(
                torch_module.nn.functional.normalize(image_repr, dim=-1).cpu().numpy()
            )
            rna_embeddings.append(
                torch_module.nn.functional.normalize(rna_repr, dim=-1).cpu().numpy()
            )
            sample_ids.extend(batch["sample_id"])

    image_df = pd.DataFrame(np.vstack(image_embeddings), index=sample_ids)
    rna_df = pd.DataFrame(np.vstack(rna_embeddings), index=sample_ids)

    if save:
        processed_dir = config.paths.processed_dir
        processed_dir.mkdir(parents=True, exist_ok=True)
        image_path = processed_dir / "image_embeddings.parquet"
        rna_path = processed_dir / "rna_embeddings.parquet"
        image_df.to_parquet(image_path)
        rna_df.to_parquet(rna_path)
        LOGGER.info("Embeddings saved to %s and %s", image_path, rna_path)

    return image_df, rna_df


def load_embeddings_from_disk(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    image_path = data_dir / "processed" / "image_embeddings.parquet"
    rna_path = data_dir / "processed" / "rna_embeddings.parquet"
    if not image_path.exists() or not rna_path.exists():
        raise FileNotFoundError(
            "Embeddings not found. Run 'biofm embed' before evaluation."
        )
    return pd.read_parquet(image_path), pd.read_parquet(rna_path)


__all__ = ["export_embeddings", "load_embeddings_from_disk"]
