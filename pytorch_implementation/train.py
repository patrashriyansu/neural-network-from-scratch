"""
PyTorch Training and Framework Comparison

Trains a PyTorch neural network on the breast cancer dataset using the same
preprocessing as the NumPy implementation, then compares outputs for parity.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pytorch_implementation.model import TorchNeuralNetwork, load_numpy_weights


def load_data():
    """Load and preprocess breast cancer data identically to the NumPy implementation."""
    data = load_breast_cancer()
    X, y = data.data, data.target.reshape(-1, 1).astype(np.float64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def train_pytorch(epochs=1000, lr=0.01):
    """
    Train PyTorch neural network on breast cancer dataset using BCELoss + SGD.

    Args:
        epochs: number of training epochs
        lr: learning rate

    Returns:
        model, (X_test, y_test) for downstream use
    """
    print("=" * 60)
    print("PyTorch Neural Network Training")
    print("=" * 60)

    X_train, X_test, y_train, y_test = load_data()

    # Convert to float32 tensors for standard training
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    model = TorchNeuralNetwork()
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, y_train_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 200 == 0:
            with torch.no_grad():
                preds = (model(X_train_t) >= 0.5).float()
                acc = (preds == y_train_t).float().mean().item()
            print(f"  Epoch {epoch+1:4d}/{epochs} | Loss: {loss.item():.4f} | Train Acc: {acc:.4f}")

    # Final evaluation on test set
    model.eval()
    with torch.no_grad():
        test_preds = (model(X_test_t) >= 0.5).float()
        test_acc = (test_preds == y_test_t).float().mean().item()

    print(f"\nFinal Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print("=" * 60)
    return model, (X_test, y_test)


def compare_with_numpy(numpy_params_path="results/numpy_model_params.npz"):
    """
    Load saved NumPy weights and compare forward pass outputs with PyTorch.

    Checks max absolute difference between numpy and pytorch outputs.
    Saves a full comparison report to results/framework_comparison.txt.
    Prints PASS if max_diff < 1e-6, FAIL otherwise.

    Args:
        numpy_params_path: path to .npz file saved by numpy implementation
    """
    print("\n" + "=" * 60)
    print("Framework Parity Comparison (NumPy vs PyTorch)")
    print("=" * 60)

    if not os.path.exists(numpy_params_path):
        msg = (
            f"[SKIP] NumPy params file not found at '{numpy_params_path}'.\n"
            "Run numpy_implementation/train.py first to generate it."
        )
        print(msg)
        return

    # Load saved numpy params
    numpy_params = np.load(numpy_params_path)

    # Load same test data
    X_train, X_test, y_train, y_test = load_data()

    # Build numpy model and set params for reference forward pass
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from numpy_implementation.model import NeuralNetwork

    np_model = NeuralNetwork(layer_sizes=[30, 16, 8, 1], seed=42)
    np_model.set_params(dict(numpy_params))
    np_output = np_model.forward(X_test)  # shape (n, 1)

    # Build PyTorch model in float64 for numeric parity
    torch_model = TorchNeuralNetwork().double()  # cast all params to float64
    load_numpy_weights(torch_model, numpy_params)

    torch_model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float64)
        torch_output = torch_model(X_test_t).numpy()  # shape (n, 1)

    # Compute differences
    abs_diff = np.abs(np_output - torch_output)
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()
    result = "PASS" if max_diff < 1e-6 else "FAIL"

    print(f"Max absolute difference:  {max_diff:.2e}")
    print(f"Mean absolute difference: {mean_diff:.2e}")
    print(f"Parity result:            {result} (threshold: 1e-6)")

    # Save full report
    os.makedirs("results", exist_ok=True)
    report_lines = [
        "Framework Parity Comparison Report",
        "=" * 50,
        f"NumPy params file: {numpy_params_path}",
        f"Test samples:      {X_test.shape[0]}",
        "",
        f"Max absolute diff:  {max_diff:.6e}",
        f"Mean absolute diff: {mean_diff:.6e}",
        f"Result:             {result}  (threshold < 1e-6)",
        "",
        "Per-sample differences (first 20):",
    ]
    for i, d in enumerate(abs_diff[:20]):
        report_lines.append(f"  Sample {i:3d}: numpy={np_output[i,0]:.8f}  pytorch={torch_output[i,0]:.8f}  diff={d[0]:.2e}")

    report_text = "\n".join(report_lines) + "\n"
    report_path = "results/framework_comparison.txt"
    with open(report_path, "w") as f:
        f.write(report_text)

    print(f"\nFull report saved to: {report_path}")
    print("=" * 60)
    return max_diff, result


if __name__ == "__main__":
    train_pytorch(epochs=1000, lr=0.01)
    compare_with_numpy()
