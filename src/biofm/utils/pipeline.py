"""Shared helpers for orchestrating BioFM pipelines."""

from __future__ import annotations

import logging

from biofm.configuration import BioFMConfig
from biofm.dataio.clinical import load_clinical_records
from biofm.dataio.microscopy import discover_microscopy_samples
from biofm.dataio.scrna import discover_scrna_samples
from biofm.dataio.toydata import generate_toy_dataset
from biofm.datamodels import (
    DatasetBundle,
    build_dataset_bundle,
    validate_dataset_counts,
)
from biofm.models.clip import BioFMClipModel
from biofm.models.encoders import create_image_encoder, create_rna_encoder

LOGGER = logging.getLogger(__name__)


def ensure_data(config: BioFMConfig, *, logger: logging.Logger | None = None) -> None:
    """Ensure the documented data layout exists, generating toy data when requested."""

    logger = logger or LOGGER
    microscopy_dir = config.paths.raw_dir / "microscopy"
    scrna_dir = config.paths.raw_dir / "scrna"
    clinical_dir = config.paths.raw_dir / "clinical"
    clinical_csv = clinical_dir / "clinical.csv"

    if (
        microscopy_dir.exists()
        and any(microscopy_dir.iterdir())
        and scrna_dir.exists()
        and any(scrna_dir.iterdir())
        and clinical_csv.exists()
    ):
        return

    if config.data.profile != "toy":
        missing = [
            str(path.relative_to(config.paths.data_dir))
            for path in (microscopy_dir, scrna_dir, clinical_csv)
            if not path.exists()
        ]
        raise FileNotFoundError(
            "Missing required data directories: " + ", ".join(missing)
        )

    samples = config.data.samples or 32
    logger.info(
        "Toy mode: generating synthetic dataset (%s samples) under %s",
        samples,
        config.paths.data_dir,
    )
    generate_toy_dataset(
        data_dir=config.paths.data_dir,
        n_samples=samples,
        image_size=config.data.image_size,
        n_genes=config.data.n_genes,
        seed=config.project.seed,
    )


def load_bundle(
    config: BioFMConfig, *, logger: logging.Logger | None = None
) -> DatasetBundle:
    """Load and validate the multimodal dataset bundle."""

    logger = logger or LOGGER
    microscopy = discover_microscopy_samples(config.paths.raw_dir / "microscopy")
    try:
        scrna = discover_scrna_samples(config.paths.raw_dir / "scrna")
    except RuntimeError as exc:
        raise RuntimeError(f"Failed to load scRNA modality: {exc}") from exc
    clinical_csv = config.paths.raw_dir / "clinical" / "clinical.csv"
    if not clinical_csv.exists():
        raise FileNotFoundError(f"Clinical labels missing at {clinical_csv}")
    clinical = load_clinical_records(clinical_csv)
    try:
        bundle = build_dataset_bundle(
            microscopy=microscopy, scrna=scrna, clinical=clinical
        )
    except ValueError as exc:  # pragma: no cover - rewrap for CLI messaging
        raise ValueError(f"Dataset validation failed: {exc}") from exc
    summary = validate_dataset_counts(bundle)
    logger.info("Loaded bundle: %s", summary)
    return bundle


def build_model(config: BioFMConfig, rna_input_dim: int) -> BioFMClipModel:
    """Instantiate the multimodal CLIP-style model from configuration."""

    embedding_dim = int(config.train.embedding_dim)
    image_encoder = create_image_encoder(
        embedding_dim=embedding_dim,
        pretrained=bool(config.train.pretrained),
    )
    rna_encoder = create_rna_encoder(
        input_dim=rna_input_dim,
        embedding_dim=embedding_dim,
        hidden_dim=int(config.train.rna_hidden_dim),
        dropout=float(config.train.dropout),
    )
    return BioFMClipModel(
        image_encoder=image_encoder,
        rna_encoder=rna_encoder,
        projector_dim=embedding_dim,
        temperature=float(config.train.temperature),
    )


__all__ = ["ensure_data", "load_bundle", "build_model"]
