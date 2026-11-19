"""
Framework parity test: NumPy vs PyTorch neural network.

Verifies that loading the same weights into both frameworks produces
numerically identical outputs (max absolute diff < 1e-5).
"""

import sys
import os
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from numpy_implementation.model import NeuralNetwork
from pytorch_implementation.model import TorchNeuralNetwork, load_numpy_weights


def test_parity():
    """
    Forward pass through numpy and pytorch (same weights, same input) must agree
    to within 1e-5 absolute difference.
    """
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 30))

    # NumPy model (seed=42 gives deterministic He init)
    np_model = NeuralNetwork(layer_sizes=[30, 16, 8, 1], seed=42)
    np_output = np_model.forward(X)         # shape (20, 1), float64

    # PyTorch model in float64
    torch_model = TorchNeuralNetwork().double()
    numpy_params = np_model.get_params()
    load_numpy_weights(torch_model, numpy_params)

    torch_model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float64)
        torch_output = torch_model(X_t).numpy()   # shape (20, 1)

    max_diff = np.abs(np_output - torch_output).max()
    assert max_diff < 1e-5, (
        f"Framework parity FAILED: max absolute diff = {max_diff:.2e} (threshold: 1e-5)"
    )
