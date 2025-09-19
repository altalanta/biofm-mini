#!/usr/bin/env python
"""Prepare toy or user-provided datasets into the standard layout."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from biofm.dataio.clinical import clinical_summary
from biofm.dataio.microscopy import discover_microscopy_samples
from biofm.dataio.scrna import discover_scrna_samples
from biofm.dataio.toydata import generate_toy_dataset
from biofm.datamodels import build_dataset_bundle, validate_dataset_counts

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger("biofm.prep")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["toy", "real"],
        default="toy",
        help="Generate synthetic data or validate an existing dataset",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data"),
        help="Root directory containing raw/{microscopy,scrna,clinical}",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=16,
        help="Number of toy samples to generate",
    )
    return parser.parse_args()


def build_from_real(input_dir: Path) -> None:
    microscopy = discover_microscopy_samples(input_dir / "raw" / "microscopy")
    scrna = discover_scrna_samples(input_dir / "raw" / "scrna")
    clinical_csv = input_dir / "raw" / "clinical" / "clinical.csv"
    if not clinical_csv.exists():
        raise FileNotFoundError(f"Expected clinical CSV at {clinical_csv}")
    from biofm.dataio.clinical import load_clinical_records

    clinical = load_clinical_records(clinical_csv)
    bundle = build_dataset_bundle(microscopy=microscopy, scrna=scrna, clinical=clinical)
    summary = validate_dataset_counts(bundle)
    LOGGER.info("Validated dataset: %s", summary)
    LOGGER.info("Clinical summary: %s", clinical_summary(clinical))


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    data_root = root / "data"
    if args.mode == "toy":
        bundle = generate_toy_dataset(root, n_samples=args.samples)
        summary = validate_dataset_counts(bundle)
        LOGGER.info("Toy dataset ready: %s", summary)
    else:
        build_from_real(args.input_dir)
        LOGGER.info("Real dataset appears valid; ready for training")


if __name__ == "__main__":
    main()
