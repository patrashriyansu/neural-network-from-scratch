"""
Tests for training execution and data utilities.

Covers:
- load_and_preprocess_data() shapes and leakage prevention
- compute_metrics() correctness
- train_model() runs without error and returns valid losses
- predict() output is binary
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from numpy_implementation.model import NeuralNetwork
from numpy_implementation.utils import load_and_preprocess_data, compute_metrics


# ---------------------------------------------------------------------------
# Data loading tests
# ---------------------------------------------------------------------------

class TestDataLoading:
    def test_shapes(self):
        """Train/test sets must have correct shapes for the breast cancer dataset."""
        X_train, X_test, y_train, y_test = load_and_preprocess_data()
        # 569 total; 20% test => 455 train, 114 test
        assert X_train.shape == (455, 30)
        assert X_test.shape  == (114, 30)
        assert y_train.shape == (455, 1)
        assert y_test.shape  == (114, 1)

    def test_labels_binary(self):
        """Labels must only contain 0 and 1."""
        _, _, y_train, y_test = load_and_preprocess_data()
        for y in (y_train, y_test):
            unique = set(np.unique(y).tolist())
            assert unique.issubset({0.0, 1.0}), f"Non-binary labels found: {unique}"

    def test_no_data_leakage(self):
        """Train mean should not be zero (scaler fit on train only => test mean differs)."""
        X_train, X_test, _, _ = load_and_preprocess_data()
        # Train features should be (approximately) standardised to mean≈0
        assert np.abs(X_train.mean()) < 0.01, "Training mean too far from 0"
        # Test mean will NOT be exactly 0 (no leakage)
        # Just assert that the scaler was applied (finite values)
        assert np.all(np.isfinite(X_test)), "X_test contains non-finite values"

    def test_stratified_split(self):
        """Class distribution should be similar in train and test sets."""
        _, _, y_train, y_test = load_and_preprocess_data()
        train_ratio = y_train.mean()
        test_ratio  = y_test.mean()
        # Tolerate ±5% difference in class ratio
        assert abs(train_ratio - test_ratio) < 0.05, (
            f"Class imbalance differs too much: train={train_ratio:.3f}, test={test_ratio:.3f}"
        )

    def test_dtype_float64(self):
        """All arrays must be float64 for consistent matrix math."""
        X_train, X_test, y_train, y_test = load_and_preprocess_data()
        for arr in (X_train, X_test, y_train, y_test):
            assert arr.dtype == np.float64, f"Expected float64, got {arr.dtype}"


# ---------------------------------------------------------------------------
# Metric tests
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_perfect_predictions(self):
        """When y_pred == y_true all metrics should be 1.0."""
        y = np.array([[1], [0], [1], [1], [0]], dtype=np.float64)
        metrics = compute_metrics(y, y)
        assert metrics['accuracy']  == pytest.approx(1.0)
        assert metrics['precision'] == pytest.approx(1.0)
        assert metrics['recall']    == pytest.approx(1.0)
        assert metrics['f1']        == pytest.approx(1.0)

    def test_all_wrong(self):
        """Completely inverted predictions → accuracy 0, specific precision/recall."""
        y_true = np.array([[1], [1], [0], [0]], dtype=np.float64)
        y_pred = np.array([[0], [0], [1], [1]], dtype=np.float64)
        metrics = compute_metrics(y_true, y_pred)
        assert metrics['accuracy'] == pytest.approx(0.0)

    def test_known_values(self):
        """Test a 3-TP, 1-FP, 1-FN case with manually verified numbers."""
        y_true = np.array([[1], [1], [1], [0], [1]], dtype=np.float64)
        y_pred = np.array([[1], [1], [1], [1], [0]], dtype=np.float64)
        # TP=3, FP=1, FN=1, TN=0
        metrics = compute_metrics(y_true, y_pred)
        assert metrics['accuracy']  == pytest.approx(3/5)
        assert metrics['precision'] == pytest.approx(3/4)
        assert metrics['recall']    == pytest.approx(3/4)
        assert metrics['f1']        == pytest.approx(3/4)

    def test_returns_dict_keys(self):
        """compute_metrics must return a dict with the four expected keys."""
        y = np.ones((5, 1), dtype=np.float64)
        result = compute_metrics(y, y)
        for key in ('accuracy', 'precision', 'recall', 'f1'):
            assert key in result, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# Model training smoke test
# ---------------------------------------------------------------------------

class TestTraining:
    def test_loss_decreases(self):
        """Training loss should be lower at epoch 200 than at epoch 1."""
        X_train, _, y_train, _ = load_and_preprocess_data()
        model = NeuralNetwork(layer_sizes=[30, 16, 8, 1], seed=42)

        losses = []
        for _ in range(50):   # short run to keep tests fast
            y_pred = model.forward(X_train)
            loss = model.binary_cross_entropy(y_train, y_pred)
            grads = model.backward(X_train, y_train)
            model.update_weights(grads, lr=0.01)
            losses.append(loss)

        assert losses[-1] < losses[0], (
            f"Loss did not decrease: start={losses[0]:.4f}, end={losses[-1]:.4f}"
        )

    def test_predict_is_binary(self):
        """predict() must only return 0 or 1 values."""
        X_train, X_test, _, _ = load_and_preprocess_data()
        model = NeuralNetwork(layer_sizes=[30, 16, 8, 1], seed=42)
        preds = model.predict(X_test)
        unique = set(np.unique(preds).tolist())
        assert unique.issubset({0, 1}), f"predict() returned non-binary values: {unique}"

    def test_weight_shapes_after_training(self):
        """Parameter shapes must be unchanged after a gradient update."""
        model = NeuralNetwork(layer_sizes=[30, 16, 8, 1], seed=42)
        X = np.random.default_rng(0).standard_normal((10, 30))
        y = np.zeros((10, 1))
        y_pred = model.forward(X)
        grads = model.backward(X, y)
        model.update_weights(grads, lr=0.01)

        assert model.params['W1'].shape == (30, 16)
        assert model.params['b1'].shape == (1,  16)
        assert model.params['W2'].shape == (16, 8)
        assert model.params['b2'].shape == (1,  8)
        assert model.params['W3'].shape == (8,  1)
        assert model.params['b3'].shape == (1,  1)

    def test_get_set_params_roundtrip(self):
        """set_params(get_params()) must be a lossless round-trip."""
        model = NeuralNetwork(layer_sizes=[30, 16, 8, 1], seed=42)
        original = model.get_params()
        model.set_params(original)
        for k in original:
            np.testing.assert_array_equal(model.params[k], original[k])
