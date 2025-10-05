"""Evaluation utilities for biofm."""

from biofm.eval.baselines import compute_baselines
from biofm.eval.linear_probe import fit_linear_probe
from biofm.eval.metrics import (
    bootstrap_classification_metrics,
    compute_auprc,
    compute_auroc,
    compute_ece,
    summarise_metrics,
)
from biofm.eval.plots import plot_calibration_curve
from biofm.eval.retrieval import compute_retrieval_metrics, save_retrieval_artifacts
from biofm.eval.slices import generate_slice_artifacts
from biofm.eval.splits import (
    load_split_manifest,
    make_grouped_splits,
    save_split_manifest,
)

__all__ = [
    "fit_linear_probe",
    "bootstrap_classification_metrics",
    "compute_auprc",
    "compute_auroc",
    "compute_ece",
    "summarise_metrics",
    "plot_calibration_curve",
    "make_grouped_splits",
    "save_split_manifest",
    "load_split_manifest",
    "generate_slice_artifacts",
    "compute_retrieval_metrics",
    "save_retrieval_artifacts",
    "compute_baselines",
]
