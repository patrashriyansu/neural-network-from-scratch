"""
Unit tests for mathematical operations in the NumPy neural network.

Covers:
- Sigmoid numerics and derivative
- ReLU derivative
- BCE loss edge cases
- Backprop gradient shapes
- He initialisation scale
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from numpy_implementation.model import NeuralNetwork


@pytest.fixture
def model():
    return NeuralNetwork(layer_sizes=[30, 16, 8, 1], seed=0)


# ---------------------------------------------------------------------------
# Sigmoid
# ---------------------------------------------------------------------------

class TestSigmoid:
    def test_sigmoid_zero(self, model):
        """σ(0) must equal 0.5."""
        result = model.sigmoid(np.array([0.0]))
        assert result == pytest.approx(0.5)

    def test_sigmoid_large_positive(self, model):
        """σ(large positive) must be very close to 1."""
        result = model.sigmoid(np.array([100.0]))
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_sigmoid_large_negative(self, model):
        """σ(large negative) must be very close to 0."""
        result = model.sigmoid(np.array([-100.0]))
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_sigmoid_monotone(self, model):
        """Sigmoid must be monotonically increasing."""
        z = np.linspace(-5, 5, 100)
        s = model.sigmoid(z)
        assert np.all(np.diff(s) > 0), "Sigmoid is not monotonically increasing"

    def test_sigmoid_derivative(self, model):
        """σ'(z) must equal σ(z)(1 - σ(z))."""
        z = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        s = model.sigmoid(z)
        expected = s * (1 - s)
        got = model.sigmoid_derivative(z)
        np.testing.assert_allclose(got, expected, rtol=1e-12)

    def test_sigmoid_output_in_range(self, model):
        """Sigmoid output must always be in (0, 1) for finite inputs."""
        z = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])
        s = model.sigmoid(z)
        assert np.all(s >= 0.0) and np.all(s <= 1.0)


# ---------------------------------------------------------------------------
# ReLU
# ---------------------------------------------------------------------------

class TestReLU:
    def test_relu_negative(self, model):
        """ReLU must return 0 for all negative inputs."""
        z = np.array([-5.0, -1.0, -0.001])
        np.testing.assert_array_equal(model.relu(z), np.zeros(3))

    def test_relu_positive(self, model):
        """ReLU must be identity for positive inputs."""
        z = np.array([0.001, 1.0, 100.0])
        np.testing.assert_array_equal(model.relu(z), z)

    def test_relu_derivative_negative(self, model):
        """ReLU derivative must be 0 for negative pre-activations."""
        z = np.array([-3.0, -1.0])
        np.testing.assert_array_equal(model.relu_derivative(z), np.zeros(2))

    def test_relu_derivative_positive(self, model):
        """ReLU derivative must be 1 for positive pre-activations."""
        z = np.array([0.1, 2.0, 100.0])
        np.testing.assert_array_equal(model.relu_derivative(z), np.ones(3))

    def test_relu_derivative_dtype(self, model):
        """ReLU derivative must return float64."""
        z = np.array([-1.0, 1.0])
        result = model.relu_derivative(z)
        assert result.dtype == np.float64


# ---------------------------------------------------------------------------
# BCE Loss
# ---------------------------------------------------------------------------

class TestBCELoss:
    def test_loss_is_scalar(self, model):
        """BCE loss must return a Python float (scalar)."""
        y = np.array([[0], [1]], dtype=np.float64)
        p = np.array([[0.3], [0.7]], dtype=np.float64)
        loss = model.binary_cross_entropy(y, p)
        assert isinstance(loss, float)

    def test_loss_decreases_toward_correct(self, model):
        """Loss should be lower when predictions are closer to truth."""
        y = np.array([[1.0], [1.0]])
        p_good = np.array([[0.9], [0.95]])
        p_bad  = np.array([[0.1], [0.05]])
        assert model.binary_cross_entropy(y, p_good) < model.binary_cross_entropy(y, p_bad)

    def test_loss_non_negative(self, model):
        """BCE loss must always be ≥ 0."""
        rng = np.random.default_rng(7)
        y = rng.integers(0, 2, (20, 1)).astype(np.float64)
        p = rng.uniform(0.01, 0.99, (20, 1))
        assert model.binary_cross_entropy(y, p) >= 0.0

    def test_loss_clips_predictions(self, model):
        """BCE must handle predictions at exact 0 or 1 without inf/nan."""
        y = np.array([[1.0], [0.0]])
        p = np.array([[0.0], [1.0]])    # worst possible
        loss = model.binary_cross_entropy(y, p)
        assert np.isfinite(loss), f"BCE returned non-finite value: {loss}"


# ---------------------------------------------------------------------------
# Backprop gradient shapes
# ---------------------------------------------------------------------------

class TestBackpropShapes:
    def test_gradient_shapes(self, model):
        """All gradients must match the shapes of their corresponding parameters."""
        rng = np.random.default_rng(5)
        X = rng.standard_normal((8, 30))
        y = rng.integers(0, 2, (8, 1)).astype(np.float64)

        model.forward(X)
        grads = model.backward(X, y)

        for name in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
            assert grads[name].shape == model.params[name].shape, (
                f"Grad shape mismatch for {name}: "
                f"expected {model.params[name].shape}, got {grads[name].shape}"
            )


# ---------------------------------------------------------------------------
# Weight initialisation
# ---------------------------------------------------------------------------

class TestInitialisation:
    def test_he_init_scale(self):
        """He init standard deviation for W1 should be close to sqrt(2/30)."""
        model = NeuralNetwork(layer_sizes=[30, 16, 8, 1], seed=99)
        expected_std = np.sqrt(2.0 / 30)
        actual_std   = model.params['W1'].std()
        # Allow generous tolerance since it's a random sample
        assert abs(actual_std - expected_std) / expected_std < 0.3, (
            f"He init std too far off: expected ≈{expected_std:.4f}, got {actual_std:.4f}"
        )

    def test_biases_zero_init(self):
        """All biases must be zero at initialisation."""
        model = NeuralNetwork(layer_sizes=[30, 16, 8, 1], seed=42)
        for name in ('b1', 'b2', 'b3'):
            np.testing.assert_array_equal(model.params[name], 0.0,
                                          err_msg=f"{name} is not zero-initialised")

    def test_seed_reproducibility(self):
        """Two models with the same seed must have identical weights."""
        m1 = NeuralNetwork(seed=7)
        m2 = NeuralNetwork(seed=7)
        for k in m1.params:
            np.testing.assert_array_equal(m1.params[k], m2.params[k])

    def test_different_seeds_differ(self):
        """Models with different seeds must have different weights."""
        m1 = NeuralNetwork(seed=1)
        m2 = NeuralNetwork(seed=2)
        # Very unlikely to be equal; check at least W1
        assert not np.array_equal(m1.params['W1'], m2.params['W1'])
