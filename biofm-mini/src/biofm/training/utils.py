"""Helper utilities for training loops."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Sequence, TypeVar

import torch
from torch.utils.data import DataLoader, Dataset

from biofm.datamodels import DatasetBundle, MicroscopySample, ScrnaSample
from biofm.dataio.microscopy import MicroscopyDataset
from biofm.dataio.scrna import ScrnaDataset

LOGGER = logging.getLogger(__name__)

SampleT = TypeVar("SampleT", MicroscopySample, ScrnaSample)


class PairedDataset(Dataset):
    """Dataset that pairs image tensors with RNA vectors using shared sample ids."""

    def __init__(self, image_dataset: MicroscopyDataset, rna_dataset: ScrnaDataset) -> None:
        if image_dataset.__len__() != rna_dataset.__len__():
            raise ValueError("Image and RNA datasets must have the same length")
        if image_dataset.samples and rna_dataset.sample_ids:
            image_ids = [sample.sample_id for sample in image_dataset.samples]
            if image_ids != rna_dataset.sample_ids:
                raise ValueError("Sample ids of image and RNA datasets are misaligned")
        self.image_dataset = image_dataset
        self.rna_dataset = rna_dataset

    def __len__(self) -> int:
        return len(self.image_dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_item = self.image_dataset[index]
        rna_item = self.rna_dataset[index]
        assert image_item["sample_id"] == rna_item["sample_id"]
        return {
            "pixel_values": image_item["pixel_values"],
            "expression": rna_item["expression"],
            "sample_id": image_item["sample_id"],
        }


def _sort_samples(samples: Sequence[SampleT]) -> List[SampleT]:
    return sorted(list(samples), key=lambda sample: sample.sample_id)


def create_paired_dataloader(
    bundle: DatasetBundle,
    batch_size: int,
    image_size: int,
    num_workers: int = 0,
    augment: bool = False,
    select_hvg: int | None = None,
) -> DataLoader:
    image_dataset = MicroscopyDataset(
        samples=_sort_samples(bundle.microscopy),
        image_size=image_size,
        augment=augment,
    )
    rna_dataset = ScrnaDataset(
        samples=_sort_samples(bundle.scrna),
        select_hvg=select_hvg,
        normalise=True,
    )
    paired_dataset = PairedDataset(image_dataset=image_dataset, rna_dataset=rna_dataset)
    return DataLoader(
        paired_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: Path, epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
        },
        path,
    )
    LOGGER.info("Saved checkpoint to %s", path)


__all__ = [
    "PairedDataset",
    "create_paired_dataloader",
    "get_device",
    "save_checkpoint",
]
