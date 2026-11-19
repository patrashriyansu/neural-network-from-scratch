"""
Utility functions for data loading, preprocessing, metrics, and plotting.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving PNGs
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------------
# Data Loading and Preprocessing
# ------------------------------------------------------------------

def load_and_preprocess_data(test_size=0.2, random_state=42):
    """
    Load the sklearn breast cancer dataset and preprocess it.

    Steps:
      1. Load dataset (569 samples, 30 features, binary labels)
      2. Stratified train/test split to preserve class distribution
      3. Standardize features: fit scaler on train, transform both splits
      4. Reshape y to column vectors (n, 1) for matrix operations

    Parameters
    ----------
    test_size    : float — fraction of data for testing (default 0.2)
    random_state : int   — reproducibility seed

    Returns
    -------
    X_train : ndarray (n_train, 30)
    X_test  : ndarray (n_test,  30)
    y_train : ndarray (n_train, 1)
    y_test  : ndarray (n_test,  1)
    """
    data = load_breast_cancer()
    X, y = data.data.astype(np.float64), data.target.astype(np.float64)

    # Stratified split — preserves malignant/benign ratio in both splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Standardize: z = (x - mean) / std  — fit ONLY on training data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Reshape to column vectors (n, 1) for consistent matrix math
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    return X_train, X_test, y_train, y_test


# ------------------------------------------------------------------
# Evaluation Metrics
# ------------------------------------------------------------------

def compute_metrics(y_true, y_pred):
    """
    Compute binary classification metrics from numpy arrays.

    Parameters
    ----------
    y_true : ndarray (n, 1) — ground truth labels {0, 1}
    y_pred : ndarray (n, 1) — predicted labels {0, 1}

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    accuracy  = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    return {
        'accuracy':  float(accuracy),
        'precision': float(precision),
        'recall':    float(recall),
        'f1':        float(f1),
    }


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def plot_learning_curves(train_losses, val_losses, save_path):
    """
    Plot training and validation loss curves and save to disk.

    Parameters
    ----------
    train_losses : list of float — per-epoch training loss
    val_losses   : list of float — per-epoch validation loss
    save_path    : str — file path to save the PNG (e.g. results/curves.png)
    """
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_losses, label='Train Loss',      color='steelblue',  linewidth=1.5)
    ax.plot(epochs, val_losses,   label='Validation Loss', color='darkorange', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Binary Cross-Entropy Loss')
    ax.set_title('Learning Curves — NumPy Neural Network')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Learning curves saved to: {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path):
    """
    Plot a 2×2 confusion matrix and save to disk.

    Parameters
    ----------
    y_true    : ndarray (n, 1) — ground truth labels
    y_pred    : ndarray (n, 1) — predicted labels
    save_path : str — file path to save the PNG
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    cm = np.array([[tn, fp],
                   [fn, tp]])

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax)

    classes = ['Malignant (0)', 'Benign (1)']
    tick_marks = [0, 1]
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=14, fontweight='bold')

    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix — NumPy Neural Network')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to: {save_path}")
