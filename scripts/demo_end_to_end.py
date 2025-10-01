#!/usr/bin/env python3
"""Deterministic end-to-end demo pipeline for biofm-mini.

Creates reproducible synthetic data, trains a minimal model, evaluates it,
and generates standardized artifacts under artifacts/demo/.
"""

import json
import random
import subprocess
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Set deterministic seeds
SEED = 1337
np.random.seed(SEED)
random.seed(SEED)

# Set PyTorch determinism if available
try:
    import torch
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    # Disable CUDA for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except ImportError:
    pass

# Import project modules
from biofm.eval.metrics import compute_auprc, compute_auroc, compute_ece
from biofm.eval.plots import save_calibration_curve, save_confusion_matrix, save_pr_curve, save_roc_curve

ARTIFACTS_DIR = Path("artifacts/demo")




def load_tiny_synthetic_data():
    """Generate deterministic synthetic binary classification data with class imbalance."""
    n_samples = 1000
    n_features = 5
    
    # Create synthetic features with known relationships
    X = np.random.normal(size=(n_samples, n_features))
    
    # Define true weights that create a meaningful classification problem
    true_weights = np.array([1.2, -0.8, 0.6, 0.4, -0.3])
    
    # Generate logits with controlled noise to ensure good performance
    logits = X @ true_weights + np.random.normal(0, 0.3, size=n_samples)
    
    # Convert to probabilities and create labels with reasonable class balance
    probs = 1 / (1 + np.exp(-logits))
    # Use median split to get approximately balanced classes that meet thresholds
    threshold = np.percentile(probs, 70)  # ~30% positive class
    y = (probs > threshold).astype(int)
    
    # Create deterministic train/test split
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    train_size = 700
    train_idx, test_idx = indices[:train_size], indices[train_size:]
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    return (X_train, y_train), (X_test, y_test)


def simple_logistic_model(X_train: np.ndarray, y_train: np.ndarray):
    """Train a simple logistic regression model using sklearn for stability."""
    # Use sklearn's LogisticRegression for more realistic and stable results
    from sklearn.linear_model import LogisticRegression
    
    # Use a well-calibrated logistic regression
    model = LogisticRegression(
        C=1.0,  # Regularization strength
        random_state=SEED,
        max_iter=1000,
        solver='lbfgs'
    )
    model.fit(X_train, y_train)
    
    return model


def predict_proba(model, X: np.ndarray) -> np.ndarray:
    """Predict probabilities using trained model."""
    # Get probability of positive class
    return model.predict_proba(X)[:, 1]


def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute comprehensive classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    
    return {
        "auroc": float(compute_auroc(y_true, y_prob)),
        "auprc": float(compute_auprc(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "ece": float(compute_ece(y_true, y_prob)),
    }


def get_version_info() -> dict:
    """Get version information for reproducibility."""
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], 
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_sha = "unknown"
    
    # Try to get package version
    try:
        from biofm._version import __version__
        package_version = __version__
    except ImportError:
        package_version = "unknown"
    
    return {
        "git_sha": git_sha,
        "package_version": package_version,
        "seed": SEED,
    }


def main():
    """Run the deterministic demo pipeline."""
    print(f"Running deterministic demo with seed {SEED}")
    
    # Create artifacts directory
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load synthetic data
    print("Loading synthetic data...")
    (X_train, y_train), (X_test, y_test) = load_tiny_synthetic_data()
    
    print(f"Train set: {len(X_train)} samples, {np.mean(y_train):.1%} positive")
    print(f"Test set: {len(X_test)} samples, {np.mean(y_test):.1%} positive")
    
    # Train simple model
    print("Training simple logistic regression model...")
    model = simple_logistic_model(X_train, y_train)
    
    # Generate predictions
    print("Generating predictions...")
    y_prob = predict_proba(model, X_test)
    
    # Compute metrics
    print("Computing metrics...")
    metrics = compute_classification_metrics(y_test, y_prob)
    
    print("Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Save metrics to JSON
    metrics_file = ARTIFACTS_DIR / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_file}")
    
    # Generate plots
    print("Generating plots...")
    save_roc_curve(y_test, y_prob, ARTIFACTS_DIR / "roc_curve.png")
    save_pr_curve(y_test, y_prob, ARTIFACTS_DIR / "pr_curve.png") 
    save_calibration_curve(y_test, y_prob, ARTIFACTS_DIR / "calibration_curve.png")
    save_confusion_matrix(y_test, y_prob, ARTIFACTS_DIR / "confusion_matrix.png")
    
    # Save version info
    version_info = get_version_info()
    version_file = ARTIFACTS_DIR / "version.txt"
    with open(version_file, 'w') as f:
        for key, value in version_info.items():
            f.write(f"{key}={value}\n")
    print(f"Saved version info to {version_file}")
    
    print(f"\nDemo completed! Artifacts saved to {ARTIFACTS_DIR}")
    print("Generated files:")
    for file_path in sorted(ARTIFACTS_DIR.glob("*")):
        print(f"  {file_path.name}")


if __name__ == "__main__":
    main()