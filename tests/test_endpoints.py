"""Statistical endpoint tests for biofm-mini deterministic demo.

Tests verify that the model performance meets predefined thresholds
and that results are deterministic across runs.
"""

# Import demo functions
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from biofm.eval.metrics import compute_auprc, compute_auroc, compute_ece

sys.path.append(str(Path(__file__).parent.parent))
from scripts.demo_end_to_end import (
    SEED,
    compute_classification_metrics,
    load_tiny_synthetic_data,
    predict_proba,
    simple_logistic_model,
)

# Test configuration
TOLERANCE = 2e-3  # Tolerance for metric variations
DETERMINISM_TOLERANCE = 1e-6  # Stricter tolerance for determinism checks


class TestStatisticalEndpoints:
    """Test class for statistical endpoint verification."""
    
    def setup_method(self):
        """Set up test data and model for each test."""
        # Ensure deterministic behavior
        np.random.seed(SEED)
        
        # Load data and train model
        (self.X_train, self.y_train), (self.X_test, self.y_test) = load_tiny_synthetic_data()
        self.model = simple_logistic_model(self.X_train, self.y_train)
        self.y_prob = predict_proba(self.model, self.X_test)
        self.y_pred = (self.y_prob >= 0.5).astype(int)
        
        # Compute metrics
        self.metrics = compute_classification_metrics(self.y_test, self.y_prob)
        self.prevalence = float(np.mean(self.y_test))
    
    def test_auroc_threshold(self):
        """Test that AUROC meets minimum threshold."""
        auroc = self.metrics["auroc"]
        assert auroc >= 0.70 - TOLERANCE, f"AUROC {auroc:.4f} below threshold 0.70"
        assert auroc <= 1.0, f"AUROC {auroc:.4f} exceeds maximum value 1.0"
    
    def test_auprc_threshold(self):
        """Test that AUPRC shows meaningful uplift over prevalence."""
        auprc = self.metrics["auprc"]
        min_threshold = self.prevalence + 0.10
        assert auprc >= min_threshold - TOLERANCE, (
            f"AUPRC {auprc:.4f} below threshold {min_threshold:.4f} "
            f"(prevalence {self.prevalence:.4f} + 0.10)"
        )
        assert auprc <= 1.0, f"AUPRC {auprc:.4f} exceeds maximum value 1.0"
    
    def test_accuracy_threshold(self):
        """Test that accuracy meets minimum threshold."""
        accuracy = self.metrics["accuracy"]
        assert accuracy >= 0.70 - TOLERANCE, f"Accuracy {accuracy:.4f} below threshold 0.70"
        assert accuracy <= 1.0, f"Accuracy {accuracy:.4f} exceeds maximum value 1.0"
    
    def test_precision_threshold(self):
        """Test that precision is reasonable."""
        precision = self.metrics["precision"]
        # Precision should be better than prevalence (random baseline)
        assert precision >= self.prevalence - TOLERANCE, (
            f"Precision {precision:.4f} below prevalence baseline {self.prevalence:.4f}"
        )
        assert precision <= 1.0, f"Precision {precision:.4f} exceeds maximum value 1.0"
    
    def test_recall_threshold(self):
        """Test that recall is reasonable."""
        recall = self.metrics["recall"]
        # Recall should be at least 0.5 (better than random)
        assert recall >= 0.50 - TOLERANCE, f"Recall {recall:.4f} below threshold 0.50"
        assert recall <= 1.0, f"Recall {recall:.4f} exceeds maximum value 1.0"
    
    def test_f1_threshold(self):
        """Test that F1 score is reasonable."""
        f1 = self.metrics["f1"]
        # F1 should be better than a naive baseline
        assert f1 >= 0.60 - TOLERANCE, f"F1 score {f1:.4f} below threshold 0.60"
        assert f1 <= 1.0, f"F1 score {f1:.4f} exceeds maximum value 1.0"
    
    def test_ece_threshold(self):
        """Test that Expected Calibration Error is within acceptable range."""
        ece = self.metrics["ece"]
        assert ece <= 0.10 + TOLERANCE, f"ECE {ece:.4f} above threshold 0.10"
        assert ece >= 0.0, f"ECE {ece:.4f} below minimum value 0.0"
    
    def test_class_balance_sanity(self):
        """Test that the synthetic data has expected class distribution."""
        # Should be roughly 30% positive based on demo data generation
        assert 0.25 <= self.prevalence <= 0.35, (
            f"Prevalence {self.prevalence:.4f} outside expected range [0.25, 0.35]"
        )
    
    def test_prediction_distribution(self):
        """Test that predictions have reasonable distribution."""
        # Predicted probabilities should cover a reasonable range
        prob_min, prob_max = np.min(self.y_prob), np.max(self.y_prob)
        assert prob_min >= 0.0, f"Minimum probability {prob_min:.4f} below 0.0"
        assert prob_max <= 1.0, f"Maximum probability {prob_max:.4f} above 1.0"
        
        # Should have some diversity in predictions
        prob_std = np.std(self.y_prob)
        assert prob_std >= 0.05, f"Probability std {prob_std:.4f} too low (predictions not diverse)"


class TestDeterminism:
    """Test class for determinism verification."""
    
    def test_data_generation_determinism(self):
        """Test that data generation is deterministic across runs."""
        np.random.seed(SEED)
        (X_train1, y_train1), (X_test1, y_test1) = load_tiny_synthetic_data()
        
        np.random.seed(SEED)
        (X_train2, y_train2), (X_test2, y_test2) = load_tiny_synthetic_data()
        
        assert np.allclose(X_train1, X_train2, atol=DETERMINISM_TOLERANCE), "Training features not deterministic"
        assert np.allclose(X_test1, X_test2, atol=DETERMINISM_TOLERANCE), "Test features not deterministic"
        assert np.array_equal(y_train1, y_train2), "Training labels not deterministic"
        assert np.array_equal(y_test1, y_test2), "Test labels not deterministic"
    
    def test_model_training_determinism(self):
        """Test that model training is deterministic."""
        np.random.seed(SEED)
        (X_train, y_train), (X_test, y_test) = load_tiny_synthetic_data()
        
        model1 = simple_logistic_model(X_train, y_train)
        model2 = simple_logistic_model(X_train, y_train)
        
        # Check that model coefficients are deterministic
        assert np.allclose(model1.coef_, model2.coef_, atol=DETERMINISM_TOLERANCE), "Model coefficients not deterministic"
        assert np.allclose(model1.intercept_, model2.intercept_, atol=DETERMINISM_TOLERANCE), "Model intercept not deterministic"
    
    def test_prediction_determinism(self):
        """Test that predictions are deterministic."""
        np.random.seed(SEED)
        (X_train, y_train), (X_test, y_test) = load_tiny_synthetic_data()
        model = simple_logistic_model(X_train, y_train)
        
        y_prob1 = predict_proba(model, X_test)
        y_prob2 = predict_proba(model, X_test)
        
        assert np.allclose(y_prob1, y_prob2, atol=DETERMINISM_TOLERANCE), "Predictions not deterministic"
    
    def test_metrics_determinism(self):
        """Test that computed metrics are deterministic."""
        np.random.seed(SEED)
        (X_train, y_train), (X_test, y_test) = load_tiny_synthetic_data()
        model = simple_logistic_model(X_train, y_train)
        y_prob = predict_proba(model, X_test)
        
        metrics1 = compute_classification_metrics(y_test, y_prob)
        metrics2 = compute_classification_metrics(y_test, y_prob)
        
        for metric_name in metrics1.keys():
            assert abs(metrics1[metric_name] - metrics2[metric_name]) < DETERMINISM_TOLERANCE, (
                f"Metric {metric_name} not deterministic: {metrics1[metric_name]} vs {metrics2[metric_name]}"
            )


class TestMetricSanity:
    """Test class for basic metric sanity checks."""
    
    def test_metric_ranges(self):
        """Test that all metrics fall within expected ranges."""
        np.random.seed(SEED)
        (X_train, y_train), (X_test, y_test) = load_tiny_synthetic_data()
        model = simple_logistic_model(X_train, y_train)
        y_prob = predict_proba(model, X_test)
        
        # Test individual metric computations
        auroc = compute_auroc(y_test, y_prob)
        auprc = compute_auprc(y_test, y_prob)
        ece = compute_ece(y_test, y_prob)
        
        assert 0.0 <= auroc <= 1.0, f"AUROC {auroc} out of range [0, 1]"
        assert 0.0 <= auprc <= 1.0, f"AUPRC {auprc} out of range [0, 1]"
        assert 0.0 <= ece <= 1.0, f"ECE {ece} out of range [0, 1]"
        
        # Test sklearn metrics
        y_pred = (y_prob >= 0.5).astype(int)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        assert 0.0 <= acc <= 1.0, f"Accuracy {acc} out of range [0, 1]"
        assert 0.0 <= prec <= 1.0, f"Precision {prec} out of range [0, 1]"
        assert 0.0 <= rec <= 1.0, f"Recall {rec} out of range [0, 1]"
        assert 0.0 <= f1 <= 1.0, f"F1 {f1} out of range [0, 1]"
    
    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_prob_perfect = np.array([0.0, 0.0, 1.0, 1.0])
        
        auroc_perfect = compute_auroc(y_true, y_prob_perfect)
        auprc_perfect = compute_auprc(y_true, y_prob_perfect)
        ece_perfect = compute_ece(y_true, y_prob_perfect)
        
        assert abs(auroc_perfect - 1.0) < 1e-10, f"Perfect AUROC should be 1.0, got {auroc_perfect}"
        assert abs(auprc_perfect - 1.0) < 1e-10, f"Perfect AUPRC should be 1.0, got {auprc_perfect}"
        assert abs(ece_perfect - 0.0) < 1e-10, f"Perfect ECE should be 0.0, got {ece_perfect}"
    
    def test_random_predictions(self):
        """Test metrics with random predictions."""
        y_true = np.array([0, 1, 0, 1, 0, 1] * 100)  # Balanced dataset
        y_prob_random = np.array([0.5] * len(y_true))  # All predictions at 0.5
        
        auroc_random = compute_auroc(y_true, y_prob_random)
        
        # AUROC should be around 0.5 for random predictions
        assert abs(auroc_random - 0.5) < 0.1, f"Random AUROC should be ~0.5, got {auroc_random}"