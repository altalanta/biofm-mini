"""Slice-based evaluation with bootstrap uncertainty intervals."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

from biofm.eval.bootstrap import bca_interval, bootstrap_statistics, jackknife_statistics
from biofm.eval.metrics import compute_auprc, compute_auroc, compute_ece

DEFAULT_BOOTSTRAP = 1000
DEFAULT_SEED = 1337


@dataclass
class SliceMetric:
    name: str
    point: float
    ci_low: float
    ci_high: float
    support: int
    delta: float
    p_value: float
    bh_adjusted: float


@dataclass
class SliceOutput:
    category: str
    value: str
    metrics: List[SliceMetric]


def _accuracy(labels: np.ndarray, scores: np.ndarray) -> float:
    preds = (scores >= 0.5).astype(int)
    return float((preds == labels).mean())


def _metric_functions(labels: np.ndarray, scores: np.ndarray) -> Dict[str, callable]:
    return {
        "auroc": lambda idx: compute_auroc(labels[idx], scores[idx]),
        "auprc": lambda idx: compute_auprc(labels[idx], scores[idx]),
        "accuracy": lambda idx: _accuracy(labels[idx], scores[idx]),
        "ece": lambda idx: compute_ece(labels[idx], scores[idx], n_bins=10),
    }


class SliceEvaluator:
    """Compute per-slice metrics with BCa intervals and BH correction."""

    def __init__(
        self,
        predictions: pd.DataFrame,
        *,
        bootstrap: int = DEFAULT_BOOTSTRAP,
        seed: int = DEFAULT_SEED,
    ) -> None:
        required = {"sample_id", "label", "score"}
        missing = required - set(predictions.columns)
        if missing:
            raise ValueError(f"Predictions missing required columns: {sorted(missing)}")
        self.data = predictions.reset_index(drop=True)
        self.bootstrap = bootstrap
        self.seed = seed
        self.labels = self.data["label"].to_numpy(dtype=int)
        self.scores = self.data["score"].to_numpy(dtype=float)

    def evaluate(
        self,
        categorical: Mapping[str, Sequence[str]],
        numeric_bins: Mapping[str, Sequence[Mapping[str, float]]],
        alpha: float,
    ) -> tuple[List[SliceOutput], Dict[str, float]]:
        global_stats = self._compute_metrics(self.labels, self.scores)
        results: List[SliceOutput] = []
        metric_pvalues: Dict[str, List[float]] = {name: [] for name in global_stats}

        data = self.data

        for column, categories in categorical.items():
            if column not in data.columns:
                continue
            values = categories or sorted(data[column].dropna().unique())
            for value in values:
                subset = data[data[column] == value]
                if subset.empty:
                    continue
                metrics, deltas, pvals = self._slice_metrics(subset, global_stats)
                for name, p in pvals.items():
                    metric_pvalues[name].append(p)
                results.append(
                    SliceOutput(
                        category=column,
                        value=str(value),
                        metrics=[
                            SliceMetric(
                                name=name,
                                point=metrics[name][0],
                                ci_low=metrics[name][1],
                                ci_high=metrics[name][2],
                                support=len(subset),
                                delta=deltas[name],
                                p_value=pvals[name],
                                bh_adjusted=0.0,
                            )
                            for name in metrics
                        ],
                    )
                )

        for column, bins in numeric_bins.items():
            if column not in data.columns:
                continue
            for spec in bins:
                min_val = spec.get("min", float("-inf"))
                max_val = spec.get("max", float("inf"))
                label = spec.get("label", f"{min_val}-{max_val}")
                mask = data[column].between(min_val, max_val, inclusive="left")
                subset = data[mask]
                if subset.empty:
                    continue
                metrics, deltas, pvals = self._slice_metrics(subset, global_stats)
                for name, p in pvals.items():
                    metric_pvalues[name].append(p)
                results.append(
                    SliceOutput(
                        category=column,
                        value=str(label),
                        metrics=[
                            SliceMetric(
                                name=name,
                                point=metrics[name][0],
                                ci_low=metrics[name][1],
                                ci_high=metrics[name][2],
                                support=len(subset),
                                delta=deltas[name],
                                p_value=pvals[name],
                                bh_adjusted=0.0,
                            )
                            for name in metrics
                        ],
                    )
                )

        for metric_name, pvalues in metric_pvalues.items():
            adjusted = benjamini_hochberg(pvalues, alpha)
            it = iter(adjusted)
            for output in results:
                for metric in output.metrics:
                    if metric.name == metric_name:
                        metric.bh_adjusted = next(it)

        global_points = {name: stats[0] for name, stats in global_stats.items()}
        return results, global_points

    def _slice_metrics(
        self,
        subset: pd.DataFrame,
        global_stats: Mapping[str, tuple[float, float, float]],
    ) -> tuple[Dict[str, tuple[float, float, float]], Dict[str, float], Dict[str, float]]:
        labels = subset["label"].to_numpy(dtype=int)
        scores = subset["score"].to_numpy(dtype=float)
        stats = self._compute_metrics(labels, scores)
        deltas = {name: stats[name][0] - global_stats[name][0] for name in stats}
        pvals = {
            name: self._two_sided_p_value(stats[name][3], global_stats[name][0])
            for name in stats
        }
        trimmed_stats = {name: (vals[0], vals[1], vals[2]) for name, vals in stats.items()}
        return trimmed_stats, deltas, pvals

    def _compute_metrics(
        self,
        labels: np.ndarray,
        scores: np.ndarray,
    ) -> Dict[str, tuple[float, float, float, np.ndarray]]:
        metric_functions = _metric_functions(labels, scores)
        rng = np.random.default_rng(self.seed)
        n = len(labels)
        indices = np.arange(n)
        results: Dict[str, tuple[float, float, float, np.ndarray]] = {}
        for name, fn in metric_functions.items():
            theta = fn(indices)
            boot = bootstrap_statistics(n, rng, fn, self.bootstrap)
            jack = jackknife_statistics(n, fn)
            ci_low, ci_high = bca_interval(theta, boot, jack)
            results[name] = (float(theta), float(ci_low), float(ci_high), boot)
        return results

    def _two_sided_p_value(self, boot: np.ndarray, reference: float) -> float:
        if boot.size == 0:
            return 1.0
        greater = np.mean(boot >= reference)
        lesser = np.mean(boot <= reference)
        return float(2 * min(greater, lesser))


def benjamini_hochberg(p_values: Sequence[float], alpha: float) -> List[float]:
    if not p_values:
        return []
    pvals = np.asarray(p_values)
    order = np.argsort(pvals)
    adjusted = np.empty_like(pvals)
    m = len(pvals)
    running = 1.0
    for rank, idx in enumerate(order, start=1):
        running = min(running, pvals[idx] * m / rank)
        adjusted[idx] = min(running, 1.0)
    return adjusted.tolist()


def generate_slice_artifacts(
    predictions: pd.DataFrame,
    metadata: pd.DataFrame,
    config: Mapping[str, object],
    output_dir: Path,
) -> Dict[str, object]:
    categorical = config.get("categorical", {})
    numeric = config.get("numeric", {})
    bootstrap = int(config.get("bootstrap", DEFAULT_BOOTSTRAP))
    seed = int(config.get("seed", DEFAULT_SEED))
    alpha = float(config.get("bh_alpha", 0.1))

    merged = predictions.merge(metadata, on="sample_id", how="left", validate="one_to_one")
    evaluator = SliceEvaluator(merged[["sample_id", "label", "score"]], bootstrap=bootstrap, seed=seed)
    results, global_metrics = evaluator.evaluate(categorical, numeric, alpha)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "global": global_metrics,
        "slices": [
            {
                "category": out.category,
                "value": out.value,
                "metrics": [metric.__dict__ for metric in out.metrics],
            }
            for out in results
        ],
    }
    (output_dir / "slices.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_slice_markdown(results, global_metrics, alpha, output_dir / "slice_report.md")

    for output in results:
        subset = merged[merged[output.category] == output.value]
        if subset.empty:
            continue
        matrix = _confusion_counts(subset)
        safe_value = str(output.value).replace("/", "-")
        path = output_dir / f"confusion_{output.category}_{safe_value}.json"
        path.write_text(json.dumps(matrix, indent=2), encoding="utf-8")

    return summary


def _write_slice_markdown(
    slices: Sequence[SliceOutput],
    global_metrics: Mapping[str, float],
    alpha: float,
    path: Path,
) -> None:
    lines = ["# Slice Evaluation", "", f"Benjamini-Hochberg α = {alpha}", ""]
    lines.append("## Global Metrics")
    for name, value in global_metrics.items():
        lines.append(f"- **{name.upper()}**: {value:.4f}")
    lines.append("")

    if not slices:
        lines.append("No slice metadata available.")
    else:
        lines.append("## Slice Summaries")
        for output in slices:
            lines.append(f"### {output.category}: {output.value}")
            lines.append("| Metric | Point | CI Low | CI High | Δ vs Global | Support | BH p-value |")
            lines.append("| --- | --- | --- | --- | --- | --- | --- |")
            for metric in output.metrics:
                lines.append(
                    f"| {metric.name} | {metric.point:.4f} | {metric.ci_low:.4f} | {metric.ci_high:.4f} "
                    f"| {metric.delta:+.4f} | {metric.support} | {metric.bh_adjusted:.4f} |"
                )
        lines.append("")

        worst = min(slices, key=lambda s: min(m.point for m in s.metrics))
        worst_metric = min(worst.metrics, key=lambda m: m.point)
        lines.append("## Worst Slice")
        lines.append(
            f"- **Slice**: {worst.category} = {worst.value}\n"
            f"- **Metric**: {worst_metric.name} ({worst_metric.point:.4f})\n"
            f"- **Support**: {worst_metric.support} samples"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def _confusion_counts(df: pd.DataFrame) -> Dict[str, int]:
    labels = df["label"].to_numpy(dtype=int)
    preds = (df["score"].to_numpy(dtype=float) >= 0.5).astype(int)
    return {
        "tp": int(((labels == 1) & (preds == 1)).sum()),
        "fp": int(((labels == 0) & (preds == 1)).sum()),
        "tn": int(((labels == 0) & (preds == 0)).sum()),
        "fn": int(((labels == 1) & (preds == 0)).sum()),
    }


__all__ = [
    "SliceEvaluator",
    "SliceMetric",
    "SliceOutput",
    "generate_slice_artifacts",
    "benjamini_hochberg",
]
