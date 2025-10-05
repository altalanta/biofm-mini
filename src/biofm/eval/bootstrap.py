"""Bootstrap utilities including BCa interval estimation."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Callable, Iterable

import numpy as np

__all__ = ["BootstrapSamples", "bca_interval", "bootstrap_statistics", "jackknife_statistics"]


@dataclass(frozen=True)
class BootstrapSamples:
    """Container for bootstrap and jackknife samples of a statistic."""

    statistic: float
    bootstrap: np.ndarray
    jackknife: np.ndarray


def bootstrap_statistics(
    n_samples: int,
    rng: np.random.Generator,
    stat_fn: Callable[[np.ndarray], float],
    n_bootstrap: int,
) -> np.ndarray:
    """Generate bootstrap replicates of a statistic."""
    values = np.empty(n_bootstrap, dtype=float)
    indices = np.arange(n_samples)
    for i in range(n_bootstrap):
        sample = rng.choice(indices, size=n_samples, replace=True)
        values[i] = stat_fn(sample)
    return values


def jackknife_statistics(n_samples: int, stat_fn: Callable[[np.ndarray], float]) -> np.ndarray:
    """Generate jackknife replicates by leave-one-out resampling."""
    values = np.empty(n_samples, dtype=float)
    indices = np.arange(n_samples)
    for i in range(n_samples):
        mask = np.delete(indices, i)
        values[i] = stat_fn(mask)
    return values


def bca_interval(
    statistic: float,
    bootstrap: Iterable[float],
    jackknife: Iterable[float],
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Compute the bias-corrected accelerated (BCa) interval.

    Args:
        statistic: Observed statistic on the original data.
        bootstrap: Bootstrap replicates of the statistic.
        jackknife: Jackknife replicates of the statistic.
        alpha: Significance level (default 0.05 -> 95% CI).

    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    boot = np.sort(np.asarray(list(bootstrap), dtype=float))
    if boot.size == 0:
        raise ValueError("Bootstrap sample is empty")
    jack = np.asarray(list(jackknife), dtype=float)
    dist = NormalDist()

    theta_hat = statistic
    z0 = dist.inv_cdf((boot < theta_hat).mean())

    jack_mean = jack.mean()
    numerator = np.sum((jack_mean - jack) ** 3)
    denominator = 6.0 * (np.sum((jack_mean - jack) ** 2) ** 1.5)
    acceleration = numerator / denominator if denominator != 0 else 0.0

    def _adjust(prob: float) -> float:
        if prob <= 0.0:
            return boot[0]
        if prob >= 1.0:
            return boot[-1]
        return np.quantile(boot, prob, method="nearest")

    alpha1 = alpha / 2
    alpha2 = 1 - alpha / 2

    def _bca_percentile(tail_alpha: float) -> float:
        z = dist.inv_cdf(tail_alpha)
        denom = 1 - acceleration * (z0 + z)
        if denom == 0:
            denom = 1e-12
        adj = dist.cdf(z0 + (z0 + z) / denom)
        return _adjust(adj)

    return _bca_percentile(alpha1), _bca_percentile(alpha2)
