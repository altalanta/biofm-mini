"""Microscopy data loading utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from biofm.datamodels import MicroscopySample

LOGGER = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = {".png", ".tif", ".tiff"}


class MicroscopyDataset(Dataset):
    """Torch dataset wrapping microscopy tiles."""

    def __init__(
        self,
        samples: Sequence[MicroscopySample],
        image_size: int = 224,
        augment: bool = False,
    ) -> None:
        self.samples = list(samples)
        base_transforms = [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
        augment_transforms: List[object] = []
        if augment:
            augment_transforms.extend(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                ]
            )
        self.transform = transforms.Compose(augment_transforms + base_transforms)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
        tensor = self.transform(image)
        return {"pixel_values": tensor, "sample_id": sample.sample_id}


def discover_microscopy_samples(directory: Path) -> List[MicroscopySample]:
    """Scan a directory for microscopy files and build schema objects."""

    samples: List[MicroscopySample] = []
    for image_path in sorted(directory.glob("*")):
        if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except OSError as exc:
            LOGGER.warning("Failed reading %s: %s", image_path, exc)
            continue
        samples.append(
            MicroscopySample(
                sample_id=image_path.stem,
                image_path=image_path,
                height=height,
                width=width,
                channels=3,
            )
        )
    if not samples:
        LOGGER.info("No microscopy samples discovered under %s", directory)
    return samples


def load_microscopy_dataset(
    directory: Path,
    image_size: int = 224,
    augment: bool = False,
) -> MicroscopyDataset:
    samples = discover_microscopy_samples(directory)
    return MicroscopyDataset(samples=samples, image_size=image_size, augment=augment)


def collate_microscopy(batch: Iterable[dict[str, torch.Tensor | str]]) -> dict[str, torch.Tensor | List[str]]:
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    sample_ids = [str(item["sample_id"]) for item in batch]
    return {"pixel_values": pixel_values, "sample_ids": sample_ids}


__all__ = [
    "MicroscopyDataset",
    "discover_microscopy_samples",
    "load_microscopy_dataset",
    "collate_microscopy",
]
