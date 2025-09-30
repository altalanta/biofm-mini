"""scRNA-seq data loading with optional Scanpy support."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from biofm.datamodels import ScrnaSample
from biofm.deps import require_torch

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import scanpy as sc  # type: ignore

    HAS_SCANPY = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_SCANPY = False


if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch
    from torch.utils.data import Dataset as TorchDataset
else:  # pragma: no cover - runtime uses duck typing
    torch = Any  # type: ignore
    TorchDataset = object


class ScrnaDataset(TorchDataset):
    """In-memory dataset of scRNA profiles aligned to samples."""

    def __init__(
        self,
        samples: Sequence[ScrnaSample],
        select_hvg: int | None = None,
        normalise: bool = True,
    ) -> None:
        self.samples = list(samples)
        self.sample_ids = [sample.sample_id for sample in self.samples]
        matrix, self.gene_names = _load_all_samples(self.samples)
        if normalise:
            matrix = _normalise_counts(matrix)
        if select_hvg is not None and select_hvg < matrix.shape[1]:
            matrix, self.gene_names = _select_hvg(matrix, self.gene_names, select_hvg)
        torch_module = require_torch()
        self.matrix = torch_module.from_numpy(matrix.astype(np.float32))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        return {
            "expression": self.matrix[index],
            "sample_id": self.sample_ids[index],
        }


def discover_scrna_samples(directory: Path) -> list[ScrnaSample]:
    """Scan for per-sample files or a single h5ad matrix."""

    samples: list[ScrnaSample] = []
    h5ad_files = list(directory.glob("*.h5ad"))
    if h5ad_files:
        if not HAS_SCANPY:
            raise RuntimeError(
                "Found .h5ad files but optional dependency 'scanpy' is not installed. "
                "Install biofm[scanpy] to enable this pathway."
            )
        samples.extend(_samples_from_h5ad(h5ad_files[0]))
    else:
        for path in sorted(directory.glob("*.csv")):
            sample_id = path.stem
            vector, _ = _read_csv_expression(path)
            library_size = int(np.sum(vector)) if vector.size else 0
            samples.append(
                ScrnaSample(
                    sample_id=sample_id,
                    expression_path=path,
                    num_genes=int(vector.size),
                    library_size=max(library_size, 1),
                    normalized=False,
                )
            )
    if not samples:
        LOGGER.info("No scRNA samples found within %s", directory)
    return samples


def _samples_from_h5ad(path: Path) -> list[ScrnaSample]:
    if not HAS_SCANPY:
        raise RuntimeError("scanpy is not installed but an .h5ad file was provided")
    adata = sc.read_h5ad(path)  # type: ignore[attr-defined]
    samples: list[ScrnaSample] = []
    n_cells = adata.n_obs
    n_genes = adata.n_vars
    for idx, sample_id in enumerate(map(str, adata.obs_names)):
        library_size = int(np.asarray(adata.X[idx]).sum())
        samples.append(
            ScrnaSample(
                sample_id=sample_id,
                expression_path=path,
                num_genes=int(n_genes),
                library_size=max(library_size, 1),
                normalized=False,
                row_index=idx,
            )
        )
    LOGGER.info("Loaded %s samples from %s", n_cells, path)
    return samples


def _load_all_samples(samples: Sequence[ScrnaSample]) -> tuple[np.ndarray, list[str]]:
    vectors: list[np.ndarray] = []
    gene_names: list[str] | None = None
    for sample in samples:
        path = sample.expression_path
        if (
            path.suffix.lower() == ".h5ad"
            and sample.row_index is not None
            and HAS_SCANPY
        ):
            vector, genes = _read_h5ad_row(path, sample.row_index)
        else:
            vector, genes = _read_csv_expression(path)
        if gene_names is None:
            gene_names = genes
        elif gene_names != genes:
            raise ValueError("Gene order mismatch between RNA samples")
        vectors.append(vector)
    if not vectors:
        return np.zeros((0, 0), dtype=np.float32), gene_names or []
    matrix = np.vstack(vectors)
    return matrix.astype(np.float32), gene_names or []


def _read_h5ad_row(path: Path, index: int) -> tuple[np.ndarray, list[str]]:
    if not HAS_SCANPY:
        raise RuntimeError("scanpy is required to read .h5ad files")
    adata = sc.read_h5ad(path)  # type: ignore[attr-defined]
    row = np.asarray(adata.X[index]).astype(np.float32).flatten()
    genes = list(map(str, adata.var_names))
    return row, genes


def _read_csv_expression(path: Path) -> tuple[np.ndarray, list[str]]:
    df = pd.read_csv(path)
    if {"gene", "value"}.issubset(df.columns):
        df = df[["gene", "value"]].copy()
        df.sort_values("gene", inplace=True)
        genes = df["gene"].astype(str).tolist()
        vector = df["value"].to_numpy(dtype=np.float32)
    else:
        df = df.set_index(df.columns[0])
        row = df.iloc[0]
        genes = row.index.astype(str).tolist()
        vector = row.to_numpy(dtype=np.float32)
    return vector, genes


def _normalise_counts(matrix: np.ndarray) -> np.ndarray:
    library = matrix.sum(axis=1, keepdims=True)
    library[library == 0] = 1.0
    matrix = matrix / library * 1e4
    return np.log1p(matrix)


def _select_hvg(
    matrix: np.ndarray, genes: list[str], k: int
) -> tuple[np.ndarray, list[str]]:
    variances = matrix.var(axis=0)
    top_idx = np.argsort(variances)[::-1][:k]
    return matrix[:, top_idx], [genes[i] for i in top_idx]


def collate_scrna(
    batch: Iterable[dict[str, torch.Tensor | str]],
) -> dict[str, torch.Tensor | list[str]]:
    torch_module = require_torch()
    expressions = torch_module.stack([item["expression"] for item in batch])
    sample_ids = [str(item["sample_id"]) for item in batch]
    return {"expression": expressions, "sample_ids": sample_ids}


__all__ = [
    "ScrnaDataset",
    "discover_scrna_samples",
    "collate_scrna",
]
