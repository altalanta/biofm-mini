"""Microscopy preprocessing helpers (tiling and normalisation)."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from PIL import Image


def normalise_tile(tile: Image.Image) -> Image.Image:
    array = np.asarray(tile).astype(np.float32)
    mean = array.mean()
    std = array.std() or 1.0
    array = (array - mean) / std
    array = np.clip((array - array.min()) / (array.max() - array.min() + 1e-6), 0, 1)
    return Image.fromarray((array * 255).astype(np.uint8))


def tile_image(image_path: Path, tile_size: int = 128, stride: int | None = None) -> List[Image.Image]:
    stride = stride or tile_size
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        tiles: List[Image.Image] = []
        for top in range(0, height - tile_size + 1, stride):
            for left in range(0, width - tile_size + 1, stride):
                box = (left, top, left + tile_size, top + tile_size)
                tiles.append(normalise_tile(image.crop(box)))
        if not tiles:
            tiles.append(normalise_tile(image.resize((tile_size, tile_size))))
    return tiles


def preprocess_and_save(image_path: Path, output_dir: Path, tile_size: int = 128) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tiles = tile_image(image_path, tile_size=tile_size)
    saved_paths = []
    for index, tile in enumerate(tiles):
        out_path = output_dir / f"{image_path.stem}_tile{index:03d}.png"
        tile.save(out_path)
        saved_paths.append(out_path)
    return saved_paths


__all__ = ["tile_image", "normalise_tile", "preprocess_and_save"]
