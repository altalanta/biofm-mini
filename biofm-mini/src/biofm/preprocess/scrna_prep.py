"""Simple scRNA-seq preprocessing utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def filter_genes(df: pd.DataFrame, min_expression: float = 0.1) -> pd.DataFrame:
    mask = (df > min_expression).sum(axis=0) > 0
    return df.loc[:, mask]


def log_normalise(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.to_numpy(dtype=float)
    library = counts.sum(axis=1, keepdims=True)
    library[library == 0] = 1.0
    norm = counts / library * 1e4
    log_norm = np.log1p(norm)
    return pd.DataFrame(log_norm, index=df.index, columns=df.columns)


def preprocess_expression(input_path: Path, output_path: Path) -> Tuple[Path, pd.DataFrame]:
    df = pd.read_csv(input_path, index_col=0)
    df = filter_genes(df)
    df = log_normalise(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    return output_path, df


__all__ = ["filter_genes", "log_normalise", "preprocess_expression"]
