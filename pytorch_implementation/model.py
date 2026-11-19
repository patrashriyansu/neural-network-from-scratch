"""
PyTorch Neural Network Model

Implements the same 30->16->8->1 architecture as the NumPy version,
using nn.Module for clean integration with PyTorch training utilities.
"""

import torch
import torch.nn as nn


class TorchNeuralNetwork(nn.Module):
    """
    Neural network with architecture: 30 -> 16 (ReLU) -> 8 (ReLU) -> 1 (Sigmoid)

    Mirrors the NumPy NeuralNetwork class for framework parity comparison.
    """

    def __init__(self):
        super(TorchNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(30, 16)   # Input -> Hidden 1
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(16, 8)    # Hidden 1 -> Hidden 2
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(8, 1)     # Hidden 2 -> Output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward pass through all layers."""
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x


def load_numpy_weights(torch_model, numpy_params):
    """
    Transfer weights from a NumPy model's saved params into a PyTorch model.

    Critical: NumPy uses weight matrices of shape (fan_in, fan_out),
    but PyTorch nn.Linear stores weights as (fan_out, fan_in).
    So we must TRANSPOSE W when loading.

    Args:
        torch_model: TorchNeuralNetwork instance
        numpy_params: dict or npz with keys W1, b1, W2, b2, W3, b3
    """
    with torch.no_grad():
        # Layer 1: W1 shape numpy (30, 16) -> PyTorch needs (16, 30)
        W1 = torch.tensor(numpy_params['W1'].T, dtype=torch.float64)
        b1 = torch.tensor(numpy_params['b1'].flatten(), dtype=torch.float64)
        torch_model.layer1.weight.copy_(W1)
        torch_model.layer1.bias.copy_(b1)

        # Layer 2: W2 shape numpy (16, 8) -> PyTorch needs (8, 16)
        W2 = torch.tensor(numpy_params['W2'].T, dtype=torch.float64)
        b2 = torch.tensor(numpy_params['b2'].flatten(), dtype=torch.float64)
        torch_model.layer2.weight.copy_(W2)
        torch_model.layer2.bias.copy_(b2)

        # Layer 3: W3 shape numpy (8, 1) -> PyTorch needs (1, 8)
        W3 = torch.tensor(numpy_params['W3'].T, dtype=torch.float64)
        b3 = torch.tensor(numpy_params['b3'].flatten(), dtype=torch.float64)
        torch_model.layer3.weight.copy_(W3)
        torch_model.layer3.bias.copy_(b3)
