#!/usr/bin/env python
"""Evaluate frozen embeddings with a linear probe."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from biofm.eval.linear_probe import probe_and_report
from biofm.utils.embeddings import export_embeddings, load_embeddings_from_disk
from biofm.utils.pipeline import load_bundle
from biofm.utils.seeds import seed_everything

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger("biofm.eval")


@hydra.main(config_path="../configs", config_name="default", version_base="1.3")
def main(cfg: DictConfig) -> None:  # type: ignore[override]
    cfg.paths.project_root = hydra.utils.get_original_cwd()
    seed_everything(int(cfg.project.seed))

    data_dir = Path(cfg.paths.data_dir)
    try:
        image_df, rna_df = load_embeddings_from_disk(data_dir)
        LOGGER.info("Loaded stored embeddings")
    except FileNotFoundError:
        LOGGER.info("Embeddings not found, regenerating via export_embeddings")
        image_df, rna_df = export_embeddings(cfg, save=True)

    bundle = load_bundle(cfg)
    labels_map = {record.sample_id: record.label for record in bundle.clinical}

    common_ids = image_df.index.intersection(rna_df.index)
    features = np.hstack(
        [image_df.loc[common_ids].to_numpy(), rna_df.loc[common_ids].to_numpy()]
    )
    labels = np.array([labels_map[sample] for sample in common_ids])

    _, summary = probe_and_report(features, labels)
    LOGGER.info("%s", summary)


if __name__ == "__main__":
    main()
