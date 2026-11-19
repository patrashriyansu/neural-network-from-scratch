"""
Tests for numerical gradient checking of the NumPy neural network.

Imports from numpy_implementation.gradient_check.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from numpy_implementation.model import NeuralNetwork
from numpy_implementation.gradient_check import check_gradients


def test_gradient_check_passes():
    """
    Run gradient check on small deterministic data.
    All parameters must report PASS (relative error < 1e-5).
    """
    rng = np.random.default_rng(42)
    X_small = rng.standard_normal((5, 30))
    y_small = rng.integers(0, 2, size=(5, 1)).astype(np.float64)

    model = NeuralNetwork(layer_sizes=[30, 16, 8, 1], seed=42)

    # check_gradients returns a list of dicts with keys: name, passed, max_abs_err, max_rel_err
    results = check_gradients(model, X_small, y_small)

    failed = [r['name'] for r in results if not r['passed']]
    assert len(failed) == 0, f"Gradient check FAILED for params: {failed}"
