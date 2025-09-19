"""Shared helpers for scripts to reduce duplication."""

from __future__ import annotations

import logging
from pathlib import Path

from omegaconf import DictConfig

from biofm.dataio.clinical import load_clinical_records
from biofm.dataio.microscopy import discover_microscopy_samples
from biofm.dataio.scrna import discover_scrna_samples
from biofm.dataio.toydata import generate_toy_dataset
from biofm.datamodels import DatasetBundle, build_dataset_bundle, validate_dataset_counts
from biofm.models.clip import BioFMClipModel
from biofm.models.encoders import create_image_encoder, create_rna_encoder

LOGGER = logging.getLogger(__name__)


def ensure_data(cfg: DictConfig, project_root: Path) -> None:
    data_dir = Path(cfg.paths.data_dir)
    microscopy_dir = data_dir / "raw" / "microscopy"
    if microscopy_dir.exists() and any(microscopy_dir.iterdir()):
        return
    samples = int(cfg.data.samples) if cfg.data.samples else 16
    LOGGER.info("Preparing toy dataset with %s samples", samples)
    generate_toy_dataset(project_root, n_samples=samples)


def load_bundle(cfg: DictConfig) -> DatasetBundle:
    data_dir = Path(cfg.paths.data_dir)
    microscopy = discover_microscopy_samples(data_dir / "raw" / "microscopy")
    scrna = discover_scrna_samples(data_dir / "raw" / "scrna")
    clinical = load_clinical_records(data_dir / "raw" / "clinical" / "clinical.csv")
    bundle = build_dataset_bundle(microscopy=microscopy, scrna=scrna, clinical=clinical)
    summary = validate_dataset_counts(bundle)
    LOGGER.info("Loaded bundle: %s", summary)
    return bundle


def build_model(cfg: DictConfig, rna_input_dim: int) -> BioFMClipModel:
    embedding_dim = int(cfg.train.embedding_dim)
    image_encoder = create_image_encoder(
        embedding_dim=embedding_dim,
        pretrained=bool(cfg.train.pretrained),
    )
    rna_encoder = create_rna_encoder(
        input_dim=rna_input_dim,
        embedding_dim=embedding_dim,
        hidden_dim=int(cfg.train.rna_hidden_dim),
        dropout=float(cfg.train.dropout),
    )
    model = BioFMClipModel(
        image_encoder=image_encoder,
        rna_encoder=rna_encoder,
        projector_dim=embedding_dim,
        temperature=float(cfg.train.temperature),
    )
    return model


__all__ = ["ensure_data", "load_bundle", "build_model"]
