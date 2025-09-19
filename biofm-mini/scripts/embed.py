#!/usr/bin/env python
"""Generate embeddings for each modality and store them in data/processed."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from biofm.utils.embeddings import export_embeddings

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger("biofm.embed")


@hydra.main(config_path="../configs", config_name="default", version_base="1.3")
def main(cfg: DictConfig) -> None:  # type: ignore[override]
    cfg.paths.project_root = hydra.utils.get_original_cwd()
    export_embeddings(cfg, save=True)


if __name__ == "__main__":
    main()
