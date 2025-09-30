"""Test that evaluation metrics are within expected ranges and include bootstrap CIs."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import pytest


def test_eval_metrics_ranges(tmp_path: Path) -> None:
    """Test that 'make eval' produces metrics within expected ranges with bootstrap CIs."""
    
    # Change to the project directory
    project_dir = Path(__file__).parent.parent
    
    # Create a temporary output directory 
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run make train first to create a checkpoint
    train_result = subprocess.run(
        ["make", "train"],
        cwd=project_dir,
        env={
            **dict(subprocess.os.environ),
            "BIOFM_OUTPUT_DIR": str(output_dir),
        },
        capture_output=True,
        text=True,
        timeout=300,  # 5 minute timeout
    )
    
    if train_result.returncode != 0:
        pytest.fail(f"make train failed: {train_result.stderr}")
    
    # Run make eval 
    eval_result = subprocess.run(
        ["make", "eval"],
        cwd=project_dir,
        env={
            **dict(subprocess.os.environ),
            "BIOFM_OUTPUT_DIR": str(output_dir),
        },
        capture_output=True,
        text=True,
        timeout=180,  # 3 minute timeout
    )
    
    if eval_result.returncode != 0:
        pytest.fail(f"make eval failed: {eval_result.stderr}")
    
    # Check that reports/metrics.json exists
    reports_dir = output_dir / "reports"
    if not reports_dir.exists():
        # Try the default output location
        reports_dir = project_dir / "outputs" / "reports"
    
    if not reports_dir.exists():
        # Check for metrics.json in output dir directly
        metrics_path = output_dir / "metrics.json"
        if not metrics_path.exists():
            metrics_path = project_dir / "outputs" / "metrics.json"
    else:
        metrics_path = reports_dir / "metrics.json"
    
    assert metrics_path.exists(), f"metrics.json not found at {metrics_path}"
    
    # Load and validate metrics
    with open(metrics_path) as f:
        metrics_data = json.load(f)
    
    # Check that evaluation metrics exist
    assert "evaluation" in metrics_data, f"No evaluation section in metrics: {metrics_data.keys()}"
    eval_metrics = metrics_data["evaluation"]
    
    # Check AUROC metrics
    assert "auroc" in eval_metrics, f"No AUROC in evaluation metrics: {eval_metrics.keys()}"
    auroc = eval_metrics["auroc"]
    
    # Validate AUROC structure and ranges
    assert "point_estimate" in auroc, "AUROC missing point_estimate"
    assert "ci_low" in auroc, "AUROC missing ci_low"
    assert "ci_high" in auroc, "AUROC missing ci_high"
    
    auroc_value = auroc["point_estimate"]
    auroc_ci_low = auroc["ci_low"]
    auroc_ci_high = auroc["ci_high"]
    
    # AUROC should be in [0.4, 0.95] range
    assert 0.4 <= auroc_value <= 0.95, f"AUROC {auroc_value} not in [0.4, 0.95]"
    
    # CI bounds should be valid
    assert auroc_ci_low <= auroc_value <= auroc_ci_high, (
        f"AUROC CI bounds invalid: {auroc_ci_low} <= {auroc_value} <= {auroc_ci_high}"
    )
    
    # Check AUPRC metrics  
    assert "auprc" in eval_metrics, f"No AUPRC in evaluation metrics: {eval_metrics.keys()}"
    auprc = eval_metrics["auprc"]
    
    # Validate AUPRC structure and ranges
    assert "point_estimate" in auprc, "AUPRC missing point_estimate"
    assert "ci_low" in auprc, "AUPRC missing ci_low"
    assert "ci_high" in auprc, "AUPRC missing ci_high"
    
    auprc_value = auprc["point_estimate"]
    auprc_ci_low = auprc["ci_low"]
    auprc_ci_high = auprc["ci_high"]
    
    # AUPRC should be in [0.4, 0.95] range
    assert 0.4 <= auprc_value <= 0.95, f"AUPRC {auprc_value} not in [0.4, 0.95]"
    
    # CI bounds should be valid
    assert auprc_ci_low <= auprc_value <= auprc_ci_high, (
        f"AUPRC CI bounds invalid: {auprc_ci_low} <= {auprc_value} <= {auprc_ci_high}"
    )
    
    print(f"✓ AUROC: {auroc_value:.3f} [{auroc_ci_low:.3f}, {auroc_ci_high:.3f}]")
    print(f"✓ AUPRC: {auprc_value:.3f} [{auprc_ci_low:.3f}, {auprc_ci_high:.3f}]")


def test_bootstrap_ci_structure() -> None:
    """Test bootstrap CI structure using the existing eval metrics functions."""
    
    import numpy as np
    from biofm.eval.metrics import bootstrap_metric, compute_auroc, compute_auprc
    
    # Create synthetic data that should give reasonable metrics
    np.random.seed(42)
    n_samples = 100
    
    # Create labels with some class imbalance
    labels = np.random.binomial(1, 0.3, n_samples)  # 30% positive class
    
    # Create scores that are correlated with labels (but with noise)
    scores = labels * 0.7 + np.random.normal(0, 0.3, n_samples)
    scores = np.clip(scores, 0, 1)  # Keep in [0, 1] range
    
    # Test AUROC bootstrap
    auroc_result = bootstrap_metric(
        compute_auroc, labels, scores, n_bootstrap=100, seed=42
    )
    
    assert hasattr(auroc_result, "point_estimate")
    assert hasattr(auroc_result, "ci_low") 
    assert hasattr(auroc_result, "ci_high")
    
    # Should be reasonable values
    assert 0.5 <= auroc_result.point_estimate <= 1.0
    assert auroc_result.ci_low <= auroc_result.point_estimate <= auroc_result.ci_high
    assert auroc_result.ci_high - auroc_result.ci_low > 0  # CI should have width
    
    # Test AUPRC bootstrap
    auprc_result = bootstrap_metric(
        compute_auprc, labels, scores, n_bootstrap=100, seed=42
    )
    
    assert hasattr(auprc_result, "point_estimate")
    assert hasattr(auprc_result, "ci_low")
    assert hasattr(auprc_result, "ci_high")
    
    # Should be reasonable values
    assert 0.0 <= auprc_result.point_estimate <= 1.0
    assert auprc_result.ci_low <= auprc_result.point_estimate <= auprc_result.ci_high
    assert auprc_result.ci_high - auprc_result.ci_low > 0  # CI should have width