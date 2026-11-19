"""
Tests for NumPy Neural Network forward pass, activations, and loss.

Imports from numpy_implementation.model.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from numpy_implementation.model import NeuralNetwork


@pytest.fixture
def model():
    """Return a fresh NeuralNetwork with deterministic weights."""
    return NeuralNetwork(layer_sizes=[30, 16, 8, 1], seed=42)


@pytest.fixture
def sample_input():
    """Return a small batch of random input data (10 samples, 30 features)."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((10, 30))


def test_output_shape(model, sample_input):
    """Forward pass output must have shape (n_samples, 1)."""
    output = model.forward(sample_input)
    assert output.shape == (10, 1), f"Expected shape (10, 1), got {output.shape}"


def test_output_range(model, sample_input):
    """Sigmoid output must be strictly in [0, 1]."""
    output = model.forward(sample_input)
    assert np.all(output >= 0.0), "Output has values below 0"
    assert np.all(output <= 1.0), "Output has values above 1"


def test_relu_activation(model):
    """relu([-1, 0, 1]) must return [0, 0, 1]."""
    x = np.array([-1.0, 0.0, 1.0])
    result = model.relu(x)
    expected = np.array([0.0, 0.0, 1.0])
    np.testing.assert_array_equal(result, expected)


def test_bce_loss_positive(model, sample_input):
    """Binary cross-entropy loss must be strictly positive for valid inputs."""
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, size=(10, 1)).astype(np.float64)
    y_pred = model.forward(sample_input)
    loss = model.binary_cross_entropy(y_true, y_pred)
    assert loss > 0.0, f"Expected BCE loss > 0, got {loss}"
