#!/usr/bin/env python3
"""Standalone evaluation script that produces comprehensive evaluation artifacts.

This script loads a saved checkpoint, computes embeddings, fits a logistic probe,
computes AUROC/AUPRC with bootstrap confidence intervals, and saves evaluation
artifacts including metrics and calibration curves.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Add the src directory to the path so we can import biofm modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from biofm.configuration import load_config
from biofm.eval.metrics import summarise_metrics
from biofm.utils.embeddings import export_embeddings, load_embeddings_from_disk
from biofm.utils.pipeline import ensure_data, load_bundle
from biofm.utils.seeds import seed_everything

app = typer.Typer(help="Standalone evaluation script for BioFM model")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def create_calibration_plot(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    output_path: Path,
    n_bins: int = 10
) -> None:
    """Create and save a calibration curve plot."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping calibration plot")
        return
    
    # Compute calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins
    )
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot calibration curve
    ax.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], "k:", label="Perfect calibration")
    
    # Formatting
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Calibration curve saved to {output_path}")


def bootstrap_predictions(
    X: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 100,
    test_size: float = 0.3,
    random_state: int = 42
) -> dict[str, Any]:
    """Run bootstrap evaluation and return detailed results."""
    
    rng = np.random.RandomState(random_state)
    bootstrap_results = []
    
    for i in range(n_bootstrap):
        # Bootstrap sample
        n_samples = len(X)
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[indices]
        y_boot = y[indices]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_boot, y_boot, test_size=test_size, random_state=i, stratify=y_boot
        )
        
        try:
            # Fit logistic regression
            clf = LogisticRegression(random_state=i, max_iter=1000)
            clf.fit(X_train, y_train)
            
            # Get predictions
            y_pred_proba = clf.predict_proba(X_test)[:, 1]
            
            # Compute metrics
            from biofm.eval.metrics import compute_auroc, compute_auprc
            auroc = compute_auroc(y_test, y_pred_proba)
            auprc = compute_auprc(y_test, y_pred_proba)
            
            bootstrap_results.append({
                "iteration": i,
                "auroc": auroc,
                "auprc": auprc,
                "n_test": len(y_test),
                "n_positive": y_test.sum(),
            })
            
        except Exception as e:
            logger.warning(f"Bootstrap iteration {i} failed: {e}")
            continue
    
    if not bootstrap_results:
        raise RuntimeError("All bootstrap iterations failed")
    
    # Convert to arrays for statistics
    auroc_values = [r["auroc"] for r in bootstrap_results]
    auprc_values = [r["auprc"] for r in bootstrap_results]
    
    return {
        "n_successful_iterations": len(bootstrap_results),
        "auroc": {
            "mean": np.mean(auroc_values),
            "std": np.std(auroc_values),
            "ci_low": np.percentile(auroc_values, 2.5),
            "ci_high": np.percentile(auroc_values, 97.5),
            "values": auroc_values,
        },
        "auprc": {
            "mean": np.mean(auprc_values),
            "std": np.std(auprc_values),
            "ci_low": np.percentile(auprc_values, 2.5),
            "ci_high": np.percentile(auprc_values, 97.5),
            "values": auprc_values,
        },
        "bootstrap_details": bootstrap_results,
    }


@app.command()
def main(
    config_path: Path = typer.Option(
        None, "--config", help="Path to config file (defaults to toy profile)"
    ),
    profile: str = typer.Option(
        "toy", "--profile", help="Config profile to use", click_type=typer.Choice(["toy", "real"])
    ),
    checkpoint_path: Path = typer.Option(
        None, "--checkpoint", help="Path to checkpoint file (auto-detected if not provided)"
    ),
    output_dir: Path = typer.Option(
        Path("outputs"), "--output-dir", help="Output directory for artifacts"
    ),
    reports_dir: Path = typer.Option(
        None, "--reports-dir", help="Reports directory (defaults to output-dir/reports)"
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    n_bootstrap: int = typer.Option(100, "--n-bootstrap", help="Number of bootstrap iterations"),
) -> None:
    """Run comprehensive evaluation and generate artifacts."""
    
    # Set random seed
    seed_everything(seed)
    
    # Load configuration
    config = load_config(profile=profile, config_path=config_path)
    
    # Override output directory if provided
    if output_dir != Path("outputs"):
        config.paths.output_dir = output_dir
    
    # Setup directories
    output_dir = Path(config.paths.output_dir)
    if reports_dir is None:
        reports_dir = output_dir / "reports"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Reports directory: {reports_dir}")
    
    # Ensure data is available
    ensure_data(config)
    
    # Determine checkpoint path
    if checkpoint_path is None:
        checkpoint_path = output_dir / "checkpoints" / "last.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    logger.info(f"Using checkpoint: {checkpoint_path}")
    
    # Load or generate embeddings
    try:
        image_df, rna_df = load_embeddings_from_disk(config.paths.data_dir)
        logger.info("Loaded existing embeddings from disk")
    except FileNotFoundError:
        logger.info("Generating embeddings from checkpoint")
        from biofm.training.utils import select_device
        device = select_device()
        
        image_df, rna_df = export_embeddings(
            config=config,
            device=device,
            checkpoint=checkpoint_path,
            batch_size=int(config.eval.batch_size),
            save=True,
        )
    
    # Load bundle and prepare data
    bundle = load_bundle(config)
    labels_map = {record.sample_id: record.label for record in bundle.clinical}
    
    # Find common samples
    common_ids = image_df.index.intersection(rna_df.index)
    if common_ids.empty:
        raise ValueError("No overlapping samples found between modalities")
    
    logger.info(f"Found {len(common_ids)} common samples")
    
    # Prepare features and labels
    features = np.hstack([
        image_df.loc[common_ids].to_numpy(),
        rna_df.loc[common_ids].to_numpy(),
    ])
    labels = np.array([labels_map[sample] for sample in common_ids])
    
    logger.info(f"Feature matrix shape: {features.shape}")
    logger.info(f"Label distribution: {np.bincount(labels)}")
    
    # Fit logistic probe for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=seed, stratify=labels
    )
    
    clf = LogisticRegression(random_state=seed, max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    
    # Compute metrics with bootstrap CIs
    metrics = summarise_metrics(y_test, y_pred_proba)
    
    # Prepare metrics output
    metrics_output = {
        "auroc": {
            "point_estimate": metrics["auroc"].point_estimate,
            "ci_low": metrics["auroc"].ci_low,
            "ci_high": metrics["auroc"].ci_high,
        },
        "auprc": {
            "point_estimate": metrics["auprc"].point_estimate,
            "ci_low": metrics["auprc"].ci_low,
            "ci_high": metrics["auprc"].ci_high,
        },
        "n_test_samples": len(y_test),
        "n_positive": y_test.sum(),
        "feature_dim": features.shape[1],
        "checkpoint": str(checkpoint_path),
    }
    
    # Run detailed bootstrap analysis
    logger.info(f"Running {n_bootstrap} bootstrap iterations...")
    bootstrap_results = bootstrap_predictions(
        features, labels, n_bootstrap=n_bootstrap, random_state=seed
    )
    
    # Save metrics.json
    metrics_path = reports_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_output, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Save bootstrap.json
    bootstrap_path = reports_dir / "bootstrap.json"
    with open(bootstrap_path, "w") as f:
        json.dump(bootstrap_results, f, indent=2)
    logger.info(f"Bootstrap results saved to {bootstrap_path}")
    
    # Create calibration curve
    calibration_path = reports_dir / "calibration_curve.png"
    create_calibration_plot(y_test, y_pred_proba, calibration_path)
    
    # Print summary
    print(f"\n=== Evaluation Summary ===")
    print(f"AUROC: {metrics_output['auroc']['point_estimate']:.3f} "
          f"[{metrics_output['auroc']['ci_low']:.3f}, {metrics_output['auroc']['ci_high']:.3f}]")
    print(f"AUPRC: {metrics_output['auprc']['point_estimate']:.3f} "
          f"[{metrics_output['auprc']['ci_low']:.3f}, {metrics_output['auprc']['ci_high']:.3f}]")
    print(f"Test samples: {metrics_output['n_test_samples']}")
    print(f"Artifacts saved to: {reports_dir}")


if __name__ == "__main__":
    app()