"""
Training script for the NumPy Neural Network.
Runs full-batch gradient descent for binary classification on the breast cancer dataset.
"""

import os
import sys
import numpy as np

# Allow running as a script from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from numpy_implementation.model import NeuralNetwork
from numpy_implementation.utils import (
    load_and_preprocess_data,
    compute_metrics,
    plot_learning_curves,
    plot_confusion_matrix,
)


def train_model(epochs=1000, lr=0.01, batch_size=None):
    """
    Train the NumPy neural network using full-batch gradient descent.

    Parameters
    ----------
    epochs     : int   — number of training iterations (default 1000)
    lr         : float — learning rate (default 0.01)
    batch_size : None  — kept for API compatibility; full-batch only

    Returns
    -------
    model       : trained NeuralNetwork instance
    train_losses: list of float — training BCE per epoch
    val_losses  : list of float — validation BCE per epoch
    val_accs    : list of float — validation accuracy per epoch
    """
    # ----------------------------------------------------------------
    # 1. Data
    # ----------------------------------------------------------------
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    print(f"  Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    # ----------------------------------------------------------------
    # 2. Model
    # ----------------------------------------------------------------
    model = NeuralNetwork(layer_sizes=[30, 16, 8, 1], seed=42)
    print(f"Model architecture: {model.layer_sizes}")
    print(f"Learning rate: {lr}  |  Epochs: {epochs}  |  Full-batch GD\n")

    # ----------------------------------------------------------------
    # 3. Training Loop
    # ----------------------------------------------------------------
    train_losses = []
    val_losses   = []
    val_accs     = []

    for epoch in range(1, epochs + 1):
        # Forward pass on training data
        y_pred_train = model.forward(X_train)

        # Compute training loss
        train_loss = model.binary_cross_entropy(y_train, y_pred_train)

        # Backpropagation
        grads = model.backward(X_train, y_train)

        # Update weights
        model.update_weights(grads, lr)

        # ---- Validation ----
        y_pred_val = model.forward(X_test)
        val_loss   = model.binary_cross_entropy(y_test, y_pred_val)
        val_preds  = (y_pred_val >= 0.5).astype(np.int32)
        val_acc    = float(np.mean(val_preds == y_test))

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Print progress every 100 epochs
        if epoch % 100 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f}")

    # ----------------------------------------------------------------
    # 4. Final Evaluation
    # ----------------------------------------------------------------
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)

    # Training set metrics
    train_preds = model.predict(X_train)
    train_metrics = compute_metrics(y_train, train_preds)

    # Test set metrics
    test_preds = model.predict(X_test)
    test_metrics = compute_metrics(y_test, test_preds)

    print(f"\nTraining Set:")
    print(f"  Accuracy : {train_metrics['accuracy']:.4f}")
    print(f"  Precision: {train_metrics['precision']:.4f}")
    print(f"  Recall   : {train_metrics['recall']:.4f}")
    print(f"  F1 Score : {train_metrics['f1']:.4f}")

    print(f"\nTest Set:")
    print(f"  Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall   : {test_metrics['recall']:.4f}")
    print(f"  F1 Score : {test_metrics['f1']:.4f}")

    final_train_loss = train_losses[-1]
    final_val_loss   = val_losses[-1]
    print(f"\nFinal Train Loss: {final_train_loss:.6f}")
    print(f"Final Val   Loss: {final_val_loss:.6f}")

    # ----------------------------------------------------------------
    # 5. Save Results
    # ----------------------------------------------------------------
    os.makedirs('results', exist_ok=True)

    # Save model parameters
    params = model.get_params()
    np.savez('results/numpy_model_params.npz', **params)
    print("\nModel parameters saved to: results/numpy_model_params.npz")

    # Save plots
    plot_learning_curves(
        train_losses, val_losses,
        save_path='results/numpy_learning_curves.png'
    )
    plot_confusion_matrix(
        y_test, test_preds,
        save_path='results/numpy_confusion_matrix.png'
    )

    return model, train_losses, val_losses, val_accs


if __name__ == "__main__":
    train_model(epochs=1000, lr=0.01)
