"""Evaluation metrics including bootstrap confidence intervals."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

LOGGER = logging.getLogger(__name__)

MetricFn = Callable[[np.ndarray, np.ndarray], float]


@dataclass
class BootstrapResult:
    point_estimate: float
    ci_low: float
    ci_high: float


def compute_auroc(labels: Iterable[int], scores: Iterable[float]) -> float:
    return roc_auc_score(np.asarray(list(labels)), np.asarray(list(scores)))


def compute_auprc(labels: Iterable[int], scores: Iterable[float]) -> float:
    return average_precision_score(np.asarray(list(labels)), np.asarray(list(scores)))


def bootstrap_metric(
    metric_fn: MetricFn,
    labels: Iterable[int],
    scores: Iterable[float],
    n_bootstrap: int = 1000,
    seed: int = 7,
) -> BootstrapResult:
    labels_arr = np.asarray(list(labels))
    scores_arr = np.asarray(list(scores))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(labels_arr), len(labels_arr))
        try:
            value = metric_fn(labels_arr[indices], scores_arr[indices])
            values.append(value)
        except ValueError:
            continue
    if not values:
        raise ValueError("Bootstrap failed: metric undefined for sampled folds")
    values_arr = np.asarray(values)
    point = metric_fn(labels_arr, scores_arr)
    low, high = np.percentile(values_arr, [2.5, 97.5])
    return BootstrapResult(point_estimate=point, ci_low=low, ci_high=high)


def summarise_metrics(
    labels: Iterable[int], scores: Iterable[float]
) -> dict[str, BootstrapResult]:
    results = {
        "auroc": bootstrap_metric(compute_auroc, labels, scores),
        "auprc": bootstrap_metric(compute_auprc, labels, scores),
    }
    return results


def decision_curve_analysis(
    labels: Iterable[int],
    scores: Iterable[float],
    thresholds: Iterable[float] | None = None,
) -> pd.DataFrame:
    labels_arr = np.asarray(list(labels))
    scores_arr = np.asarray(list(scores))
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 9)
    net_benefits = []
    prevalence = labels_arr.mean()
    for threshold in thresholds:
        predictions = scores_arr >= threshold
        true_positive = np.logical_and(predictions, labels_arr == 1).sum()
        false_positive = np.logical_and(predictions, labels_arr == 0).sum()
        n = len(labels_arr)
        nb = (true_positive / n) - (false_positive / n) * (threshold / (1 - threshold))
        net_benefits.append(
            {"threshold": threshold, "net_benefit": nb, "prevalence": prevalence}
        )
    return pd.DataFrame(net_benefits)


__all__ = [
    "BootstrapResult",
    "compute_auroc",
    "compute_auprc",
    "bootstrap_metric",
    "summarise_metrics",
    "decision_curve_analysis",
]
