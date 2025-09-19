#!/usr/bin/env python
"""Train the CLIP-style multimodal model using Hydra configs."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from biofm.models.lora import apply_lora_adapters
from biofm.training.loop import LoopConfig, train_model
from biofm.training.utils import create_paired_dataloader
from biofm.utils.pipeline import build_model, ensure_data, load_bundle
from biofm.utils.seeds import seed_everything

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger("biofm.train")


@hydra.main(config_path="../configs", config_name="default", version_base="1.3")
def main(cfg: DictConfig) -> None:  # type: ignore[override]
    cfg.paths.project_root = hydra.utils.get_original_cwd()
    project_root = Path(cfg.paths.project_root)
    seed_everything(int(cfg.project.seed))

    ensure_data(cfg, project_root)
    bundle = load_bundle(cfg)

    dataloader = create_paired_dataloader(
        bundle=bundle,
        batch_size=int(cfg.train.batch_size),
        image_size=int(cfg.data.image_size),
        augment=bool(cfg.train.augment),
        select_hvg=int(cfg.data.select_hvg) if cfg.data.select_hvg else None,
    )

    rna_input_dim = dataloader.dataset.rna_dataset.matrix.shape[1]  # type: ignore[attr-defined]
    model = build_model(cfg, rna_input_dim)

    if cfg.train.get("use_lora", False):
        apply_lora_adapters(
            model,
            target_modules=["image_encoder.head", "rna_encoder.network"],
            rank=int(cfg.train.lora_rank),
            alpha=float(cfg.train.lora_alpha),
        )

    loop_cfg = LoopConfig(
        epochs=int(cfg.train.epochs),
        learning_rate=float(cfg.train.learning_rate),
        weight_decay=float(cfg.train.weight_decay),
        grad_clip=float(cfg.train.grad_clip),
        amp=bool(cfg.train.amp),
        checkpoint_dir=Path(cfg.paths.output_dir) / "checkpoints",
    )
    checkpoint = train_model(model=model, dataloader=dataloader, config=loop_cfg)
    LOGGER.info("Training complete; checkpoint saved to %s", checkpoint)


if __name__ == "__main__":
    main()
