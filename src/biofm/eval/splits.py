"""Grouped cross-validation utilities with manifest persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

try:  # pragma: no cover - optional for older scikit-learn releases
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover
    StratifiedGroupKFold = None  # type: ignore[assignment]

from biofm.datamodels import DatasetBundle

__all__ = [
    "dataframe_from_bundle",
    "make_grouped_splits",
    "save_split_manifest",
    "load_split_manifest",
]

DEFAULT_SEED = 1337


def dataframe_from_bundle(bundle: DatasetBundle, group_col: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for record in bundle.clinical:
        metadata = record.metadata or {}
        group_id = (
            metadata.get(group_col)
            or metadata.get("patient_id")
            or metadata.get("slide_id")
            or record.sample_id
        )
        rows.append(
            {
                "sample_id": record.sample_id,
                "label": record.label,
                "group_id": group_id,
            }
        )
    return pd.DataFrame(rows)


def make_grouped_splits(
    df: pd.DataFrame,
    *,
    sample_col: str = "sample_id",
    label_col: str = "label",
    group_col: str = "group_id",
    n_splits: int = 5,
    stratify: bool = True,
    random_state: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Create grouped k-fold splits with optional stratification."""

    required = {sample_col, label_col, group_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    indices = np.arange(len(df))
    labels = df[label_col].to_numpy()
    groups = df[group_col].to_numpy()

    splits: List[tuple[np.ndarray, np.ndarray]] = []
    if stratify and StratifiedGroupKFold is not None:
        try:
            splitter = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
            splits = list(splitter.split(indices, labels, groups))
        except ValueError:
            splits = []
    if not splits:
        splitter = GroupKFold(n_splits=n_splits)
        splits = list(splitter.split(indices, groups=groups))

    all_indices = np.arange(len(df))
    test_indices_per_fold = [test_idx for _, test_idx in splits]
    rows: List[Dict[str, object]] = []

    for fold, (_, test_idx) in enumerate(splits):
        if n_splits >= 3:
            val_fold = (fold + 1) % n_splits
            val_idx = test_indices_per_fold[val_fold]
        else:
            # With only two folds, derive a validation subset from the training pool.
            pool = np.setdiff1d(all_indices, test_idx, assume_unique=True)
            if pool.size == 0:
                val_idx = np.empty(0, dtype=int)
            else:
                rng = np.random.default_rng(random_state + fold)
                val_size = max(1, int(0.2 * pool.size))
                val_idx = rng.choice(pool, size=val_size, replace=False)
        train_mask = np.ones_like(all_indices, dtype=bool)
        train_mask[test_idx] = False
        train_mask[val_idx] = False
        train_idx = all_indices[train_mask]

        _append_rows(df, train_idx, sample_col, group_col, fold, "train", rows)
        _append_rows(df, val_idx, sample_col, group_col, fold, "val", rows)
        _append_rows(df, test_idx, sample_col, group_col, fold, "test", rows)

    manifest = pd.DataFrame(rows)
    manifest.sort_values(["fold", "sample_id", "split"], inplace=True)
    manifest.reset_index(drop=True, inplace=True)
    return manifest


def _append_rows(
    df: pd.DataFrame,
    idx: Iterable[int],
    sample_col: str,
    group_col: str,
    fold: int,
    split: str,
    rows: List[Dict[str, object]],
) -> None:
    for i in idx:
        rows.append(
            {
                "sample_id": df.iloc[i][sample_col],
                "group_id": df.iloc[i][group_col],
                "fold": fold,
                "split": split,
            }
        )


def save_split_manifest(manifest: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False)


def load_split_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Split manifest not found at {path}")
    manifest = pd.read_csv(path)
    expected = {"sample_id", "group_id", "fold", "split"}
    missing = expected - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
    return manifest
