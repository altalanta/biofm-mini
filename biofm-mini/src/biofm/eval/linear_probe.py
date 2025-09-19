"""Train a linear probe on frozen embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from biofm.eval.metrics import BootstrapResult, summarise_metrics


@dataclass
class ProbeResult:
    model: Pipeline
    metrics: Dict[str, BootstrapResult]


def fit_linear_probe(embeddings: np.ndarray, labels: np.ndarray, seed: int = 7) -> ProbeResult:
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array")
    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError("Embeddings and labels must align")

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    random_state=seed,
                    max_iter=200,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    pipeline.fit(embeddings, labels)
    scores = pipeline.predict_proba(embeddings)[:, 1]
    metrics = summarise_metrics(labels, scores)
    return ProbeResult(model=pipeline, metrics=metrics)


def probe_and_report(embeddings: np.ndarray, labels: np.ndarray) -> Tuple[ProbeResult, str]:
    result = fit_linear_probe(embeddings, labels)
    summary_lines = []
    for name, metric in result.metrics.items():
        summary_lines.append(
            f"{name.upper()}: {metric.point_estimate:.3f} (95% CI {metric.ci_low:.3f}-{metric.ci_high:.3f})"
        )
    return result, "\n".join(summary_lines)


__all__ = ["fit_linear_probe", "ProbeResult", "probe_and_report"]
