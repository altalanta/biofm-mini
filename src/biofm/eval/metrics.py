"""Evaluation metrics including bootstrap confidence intervals."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from biofm.eval.bootstrap import bca_interval, bootstrap_statistics, jackknife_statistics

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
    n = len(labels_arr)
    indices = np.arange(n)
    rng = np.random.default_rng(seed)

    def stat(idx: np.ndarray) -> float:
        return metric_fn(labels_arr[idx], scores_arr[idx])

    point = stat(indices)
    boot = bootstrap_statistics(n, rng, stat, n_bootstrap)
    jack = jackknife_statistics(n, stat)
    low, high = bca_interval(point, boot, jack)
    return BootstrapResult(point_estimate=point, ci_low=low, ci_high=high)


def summarise_metrics(
    labels: Iterable[int], scores: Iterable[float]
) -> dict[str, BootstrapResult]:
    results = {
        "auroc": bootstrap_metric(compute_auroc, labels, scores),
        "auprc": bootstrap_metric(compute_auprc, labels, scores),
    }
    return results


def compute_ece(labels: Iterable[int], scores: Iterable[float], n_bins: int = 10) -> float:
    labels_arr = np.asarray(list(labels))
    scores_arr = np.asarray(list(scores))
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(scores_arr, bins) - 1

    ece_val = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if not np.any(mask):
            continue
        conf = scores_arr[mask].mean()
        acc = labels_arr[mask].mean()
        weight = mask.sum() / len(labels_arr)
        ece_val += abs(acc - conf) * weight
    return float(ece_val)


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


def bootstrap_classification_metrics(
    labels: Iterable[int],
    scores: Iterable[float],
    *,
    n_bootstrap: int = 1000,
    seed: int = 1337,
) -> dict[str, dict[str, float]]:
    labels_arr = np.asarray(list(labels), dtype=int)
    scores_arr = np.asarray(list(scores), dtype=float)
    preds = (scores_arr >= 0.5).astype(int)
    n = len(labels_arr)
    indices = np.arange(n)
    rng = np.random.default_rng(seed)

    def accuracy_fn(idx: np.ndarray) -> float:
        return float((preds[idx] == labels_arr[idx]).mean())

    def ece_fn(idx: np.ndarray) -> float:
        return compute_ece(labels_arr[idx], scores_arr[idx], n_bins=10)

    metric_functions: dict[str, Callable[[np.ndarray], float]] = {
        "auroc": lambda idx: compute_auroc(labels_arr[idx], scores_arr[idx]),
        "auprc": lambda idx: compute_auprc(labels_arr[idx], scores_arr[idx]),
        "accuracy": accuracy_fn,
        "ece": ece_fn,
    }

    results: dict[str, dict[str, float]] = {}
    for name, fn in metric_functions.items():
        theta = fn(indices)
        boot = bootstrap_statistics(n, rng, fn, n_bootstrap)
        jack = jackknife_statistics(n, fn)
        ci_low, ci_high = bca_interval(theta, boot, jack)
        results[name] = {
            "point_estimate": float(theta),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
        }
    return results


__all__ = [
    "BootstrapResult",
    "compute_auroc",
    "compute_auprc",
    "compute_ece",
    "bootstrap_metric",
    "summarise_metrics",
    "decision_curve_analysis",
    "bootstrap_classification_metrics",
]
