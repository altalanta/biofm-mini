import numpy as np

from biofm.eval.metrics import bootstrap_metric, compute_auprc, compute_auroc


def test_bootstrap_metric() -> None:
    labels = np.array([0, 1, 0, 1, 0, 1])
    scores = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    auroc = compute_auroc(labels, scores)
    auprc = compute_auprc(labels, scores)
    assert 0.5 < auroc <= 1.0
    assert 0.5 < auprc <= 1.0
    result = bootstrap_metric(compute_auroc, labels, scores, n_bootstrap=32, seed=1)
    assert result.ci_low <= result.point_estimate <= result.ci_high
