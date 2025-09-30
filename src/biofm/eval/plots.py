"""Plotting utilities for evaluation metrics."""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Union

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve


def save_roc_curve(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    path: Union[str, Path],
    title: str = "ROC Curve"
) -> None:
    """Save ROC curve plot."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting")
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], '--', color='gray', alpha=0.6, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def save_pr_curve(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    path: Union[str, Path],
    title: str = "Precision-Recall Curve"
) -> None:
    """Save Precision-Recall curve plot."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting")
    
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label='PR curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def save_calibration_curve(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    path: Union[str, Path],
    n_bins: int = 10,
    title: str = "Calibration Curve"
) -> None:
    """Save calibration curve plot."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting")
    
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    
    bin_confidences = []
    bin_accuracies = []
    
    for i in range(n_bins):
        mask = bin_indices == i
        if np.any(mask):
            bin_confidences.append(y_prob[mask].mean())
            bin_accuracies.append(y_true[mask].mean())
    
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], '--', color='gray', alpha=0.6, label='Perfect calibration')
    if bin_confidences:
        plt.plot(bin_confidences, bin_accuracies, marker='o', linewidth=2, label='Model')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def save_confusion_matrix(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    path: Union[str, Path],
    threshold: float = 0.5,
    title: str = "Confusion Matrix"
) -> None:
    """Save confusion matrix plot."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for plotting")
    
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar()
    plt.title(title)
    
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), 
                    horizontalalignment='center', 
                    verticalalignment='center',
                    fontsize=14, fontweight='bold')
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks([0, 1], ['Negative', 'Positive'])
    plt.yticks([0, 1], ['Negative', 'Positive'])
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


__all__ = [
    "save_roc_curve",
    "save_pr_curve", 
    "save_calibration_curve",
    "save_confusion_matrix",
]