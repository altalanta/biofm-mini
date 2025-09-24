"""Utilities to build a tiny synthetic multimodal dataset."""

from __future__ import annotations

import logging
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from biofm.datamodels import (
    ClinicalRecord,
    DatasetBundle,
    MicroscopySample,
    ScrnaSample,
    build_dataset_bundle,
)
from biofm.utils.seeds import seed_everything

LOGGER = logging.getLogger(__name__)


def generate_toy_dataset(
    base_dir: Path,
    n_samples: int = 16,
    image_size: int = 64,
    n_genes: int = 64,
    seed: int = 7,
) -> DatasetBundle:
    """Create a paired toy dataset with simple correlations."""

    seed_everything(seed)
    microscopy_dir = base_dir / "data" / "raw" / "microscopy"
    scrna_dir = base_dir / "data" / "raw" / "scrna"
    clinical_dir = base_dir / "data" / "raw" / "clinical"
    processed_dir = base_dir / "data" / "processed"

    for directory in [microscopy_dir, scrna_dir, clinical_dir, processed_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    microscopy_samples: list[MicroscopySample] = []
    scrna_samples: list[ScrnaSample] = []
    clinical_records: list[ClinicalRecord] = []

    gene_names = [f"gene_{i:03d}" for i in range(n_genes)]

    clinical_rows = []
    for index in range(n_samples):
        sample_id = f"sample_{index:03d}"
        label = int(index % 2 == 0)

        # Create a simple blob pattern with label-specific shifts.
        image_array = _synthetic_microscopy(image_size, label, index)
        image_path = microscopy_dir / f"{sample_id}.png"
        Image.fromarray(image_array).save(image_path)

        expression = _synthetic_expression(n_genes, label)
        scrna_path = scrna_dir / f"{sample_id}.csv"
        expr_df = pd.DataFrame({"gene": gene_names, "value": expression})
        expr_df.to_csv(scrna_path, index=False)

        age = 45 + label * 5 + random.uniform(-3, 3)
        sex = random.choice(["female", "male"])
        metadata = {"set": "toy"}

        microscopy_samples.append(
            MicroscopySample(
                sample_id=sample_id,
                image_path=image_path,
                height=image_size,
                width=image_size,
                channels=3,
                staining="synthetic",
            )
        )
        scrna_samples.append(
            ScrnaSample(
                sample_id=sample_id,
                expression_path=scrna_path,
                num_genes=n_genes,
                library_size=int(expression.sum()),
                normalized=False,
            )
        )
        clinical_records.append(
            ClinicalRecord(
                sample_id=sample_id,
                label=label,
                age=age,
                sex=sex,
                metadata=metadata,
            )
        )
        clinical_rows.append(
            {
                "sample_id": sample_id,
                "label": label,
                "age": round(age, 2),
                "sex": sex,
                "cohort": "toy",
            }
        )

    clinical_df = pd.DataFrame(clinical_rows)
    clinical_csv = clinical_dir / "clinical.csv"
    clinical_df.to_csv(clinical_csv, index=False)

    LOGGER.info(
        "Generated toy dataset with %s samples (%s positive)",
        n_samples,
        sum(record.label for record in clinical_records),
    )

    return build_dataset_bundle(
        microscopy=microscopy_samples,
        scrna=scrna_samples,
        clinical=clinical_records,
    )


def _synthetic_microscopy(image_size: int, label: int, index: int) -> np.ndarray:
    x = np.linspace(-1, 1, image_size)
    xv, yv = np.meshgrid(x, x)
    angle = (index / max(1, image_size)) * math.pi
    blob = np.exp(-((xv - label * 0.3) ** 2 + (yv - label * 0.3) ** 2) * 5)
    stripes = 0.5 * (1 + np.sin(10 * (xv * math.cos(angle) + yv * math.sin(angle))))
    image = (0.6 * blob + 0.4 * stripes)
    noise = np.random.normal(scale=0.05, size=(image_size, image_size))
    image = np.clip(image + noise, 0.0, 1.0)
    rgb = np.stack([image, image * (0.7 + 0.3 * label), image[::-1, :]], axis=-1)
    return (rgb * 255).astype(np.uint8)


def _synthetic_expression(n_genes: int, label: int) -> np.ndarray:
    baseline = np.random.gamma(shape=2.0, scale=2.0, size=n_genes)
    signal = np.linspace(0.1, 1.0, n_genes)
    shift = label * 1.5 * signal
    return baseline + shift


__all__ = ["generate_toy_dataset"]
