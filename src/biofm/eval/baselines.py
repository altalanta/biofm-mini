"""Baseline classifiers to benchmark against the main model."""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from biofm.eval.metrics import compute_ece

DEFAULT_SEED = 1337


def _metric_summary(labels: np.ndarray, scores: np.ndarray) -> Mapping[str, float]:
    preds = (scores >= 0.5).astype(int)
    return {
        "auroc": float(_safe_auc(labels, scores)),
        "auprc": float(_safe_auprc(labels, scores)),
        "accuracy": float((preds == labels).mean()),
        "ece": float(compute_ece(labels, scores, n_bins=10)),
    }


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(labels)) < 2:
        return float("nan")
    return roc_auc_score(labels, scores)


def _safe_auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    if len(np.unique(labels)) < 2:
        return float("nan")
    return average_precision_score(labels, scores)


def majority_baseline(
    train_labels: np.ndarray, test_labels: np.ndarray
) -> Mapping[str, float]:
    prior = train_labels.mean() if train_labels.size else 0.5
    scores = np.full_like(test_labels, fill_value=prior, dtype=float)
    return _metric_summary(test_labels, scores)


def random_projection_baseline(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    *,
    seed: int = DEFAULT_SEED,
) -> Mapping[str, float]:
    if train_features.size == 0:
        return {"auroc": float("nan"), "auprc": float("nan"), "accuracy": float("nan"), "ece": float("nan")}
    rng = np.random.default_rng(seed)
    projection_dim = min(64, train_features.shape[1])
    projection = rng.normal(size=(train_features.shape[1], projection_dim))
    train_proj = train_features @ projection
    test_proj = test_features @ projection
    clf = LogisticRegression(max_iter=500, random_state=seed)
    clf.fit(train_proj, train_labels)
    scores = clf.predict_proba(test_proj)[:, 1]
    return _metric_summary(test_labels, scores)


def clinical_baseline(
    train_metadata: pd.DataFrame,
    train_labels: np.ndarray,
    test_metadata: pd.DataFrame,
    test_labels: np.ndarray,
) -> Mapping[str, float]:
    features_train = _clinical_features(train_metadata)
    features_test = _clinical_features(test_metadata)
    if features_train.size == 0:
        raise ValueError("No clinical features available")
    clf = LogisticRegression(max_iter=500, random_state=DEFAULT_SEED)
    clf.fit(features_train, train_labels)
    scores = clf.predict_proba(features_test)[:, 1]
    return _metric_summary(test_labels, scores)


def _clinical_features(metadata: pd.DataFrame) -> np.ndarray:
    components = []
    if "age" in metadata.columns:
        age = metadata["age"].astype(float).fillna(metadata["age"].median())
        components.append(age.to_numpy().reshape(-1, 1))
    if "sex" in metadata.columns:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoded = encoder.fit_transform(metadata[["sex"]].fillna("unknown"))
        components.append(encoded)
    return np.hstack(components) if components else np.empty((len(metadata), 0))


def compute_baselines(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    train_metadata: pd.DataFrame,
    test_metadata: pd.DataFrame,
) -> Mapping[str, Mapping[str, float]]:
    results = {
        "majority": majority_baseline(train_labels, test_labels),
        "random_projection": random_projection_baseline(
            train_features, train_labels, test_features, test_labels
        ),
    }
    try:
        results["clinical"] = clinical_baseline(
            train_metadata, train_labels, test_metadata, test_labels
        )
    except ValueError:
        pass
    return results


__all__ = [
    "compute_baselines",
    "majority_baseline",
    "random_projection_baseline",
    "clinical_baseline",
]
