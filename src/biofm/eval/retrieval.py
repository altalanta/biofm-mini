"""Cross-modal retrieval metrics with bootstrap confidence intervals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd

from biofm.eval.bootstrap import bca_interval, bootstrap_statistics, jackknife_statistics

DEFAULT_KS = (1, 5, 10)
DEFAULT_BOOTSTRAP = 1000
DEFAULT_SEED = 1337


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a_norm @ b_norm.T


def _recall_at_k(sim: np.ndarray, k: int) -> float:
    top_k = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    matches = np.any(top_k == np.arange(sim.shape[0])[:, None], axis=1)
    return float(matches.mean())


def _mean_average_precision(sim: np.ndarray) -> float:
    ranks = np.argsort(-sim, axis=1)
    positions = np.where(ranks == np.arange(sim.shape[0])[:, None])[1]
    reciprocal = 1.0 / (positions + 1)
    return float(reciprocal.mean())


def _metric_functions(sim: np.ndarray, ks: Sequence[int]) -> Dict[str, callable]:
    return {
        f"recall@{k}": lambda idx, k=k: _recall_at_k(sim[np.ix_(idx, idx)], k)
        for k in ks
    } | {"map": lambda idx: _mean_average_precision(sim[np.ix_(idx, idx)])}


def compute_retrieval_metrics(
    image_embeddings: pd.DataFrame,
    rna_embeddings: pd.DataFrame,
    *,
    ks: Sequence[int] = DEFAULT_KS,
    bootstrap: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> Mapping[str, Mapping[str, float]]:
    common = image_embeddings.index.intersection(rna_embeddings.index)
    if common.empty:
        raise ValueError("No overlapping samples for retrieval evaluation")
    image = image_embeddings.loc[common].to_numpy(dtype=float)
    rna = rna_embeddings.loc[common].to_numpy(dtype=float)

    sim_image_to_rna = _cosine_similarity(image, rna)
    sim_rna_to_image = sim_image_to_rna.T

    rng = np.random.default_rng(seed)
    metrics: Dict[str, Mapping[str, float]] = {}

    for direction, sim in {
        "image_to_rna": sim_image_to_rna,
        "rna_to_image": sim_rna_to_image,
    }.items():
        functions = _metric_functions(sim, ks)
        n = sim.shape[0]
        indices = np.arange(n)
        for name, fn in functions.items():
            theta = fn(indices)
            boot = bootstrap_statistics(n, rng, fn, bootstrap)
            jack = jackknife_statistics(n, fn)
            low, high = bca_interval(theta, boot, jack)
            metrics[f"{direction}_{name}"] = {
                "point_estimate": float(theta),
                "ci_low": float(low),
                "ci_high": float(high),
            }
    return metrics


def save_retrieval_artifacts(
    metrics: Mapping[str, Mapping[str, float]],
    output_dir: Path,
    ks: Sequence[int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "retrieval.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - plotting optional
        return

    directions = ["image_to_rna", "rna_to_image"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for direction in directions:
        recalls = [metrics[f"{direction}_recall@{k}"]["point_estimate"] for k in ks]
        ax.plot(ks, recalls, marker="o", label=direction.replace("_", " → "))
    ax.set_xlabel("K")
    ax.set_ylabel("Recall@K")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "retrieval_recall.png", dpi=150)
    plt.close(fig)


__all__ = ["compute_retrieval_metrics", "save_retrieval_artifacts"]
