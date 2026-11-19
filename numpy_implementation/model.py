"""
Neural Network Implementation using NumPy (No Autograd)
========================================================
Architecture: 30 -> 16 (ReLU) -> 8 (ReLU) -> 1 (Sigmoid)
Task: Binary classification (breast cancer dataset)

Math notation:
  Z[l] = A[l-1] @ W[l] + b[l]   (pre-activation)
  A[l] = activation(Z[l])        (post-activation)
  Loss = Binary Cross-Entropy (BCE)
"""

import numpy as np


class NeuralNetwork:
    """
    Fully-connected neural network with manual forward pass and backpropagation.

    Layer sizes: [30, 16, 8, 1]
      Layer 1: 30 inputs  -> 16 neurons, ReLU
      Layer 2: 16 neurons ->  8 neurons, ReLU
      Layer 3:  8 neurons ->  1 neuron,  Sigmoid
    """

    def __init__(self, layer_sizes=None, seed=42):
        """
        Parameters
        ----------
        layer_sizes : list of int
            Number of units in each layer including input.
            Default: [30, 16, 8, 1]
        seed : int
            Random seed for reproducibility.
        """
        if layer_sizes is None:
            layer_sizes = [30, 16, 8, 1]
        self.layer_sizes = layer_sizes
        self.seed = seed
        self.cache = {}   # stores Z and A values for backprop
        self.params = {}  # stores W and b for all layers
        self.initialize_weights()

    # ------------------------------------------------------------------
    # Weight Initialization
    # ------------------------------------------------------------------

    def initialize_weights(self):
        """
        He (Kaiming) initialization for ReLU layers:
          W ~ N(0, sqrt(2 / fan_in))

        This keeps variance stable across layers for ReLU networks.
        The last layer (sigmoid) also uses He init here for simplicity.

        Weights shape: (fan_in, fan_out)  — column-vector convention for X @ W
        """
        np.random.seed(self.seed)
        num_layers = len(self.layer_sizes) - 1  # number of weight matrices

        for l in range(1, num_layers + 1):
            fan_in = self.layer_sizes[l - 1]
            fan_out = self.layer_sizes[l]
            # He initialization: scale factor = sqrt(2 / fan_in)
            self.params[f'W{l}'] = (
                np.random.randn(fan_in, fan_out).astype(np.float64)
                * np.sqrt(2.0 / fan_in)
            )
            # Biases initialized to zero
            self.params[f'b{l}'] = np.zeros((1, fan_out), dtype=np.float64)

    # ------------------------------------------------------------------
    # Activation Functions
    # ------------------------------------------------------------------

    def relu(self, z):
        """
        ReLU activation: f(z) = max(0, z)

        Parameters
        ----------
        z : ndarray of shape (n_samples, n_units)

        Returns
        -------
        ndarray, same shape as z
        """
        return np.maximum(0.0, z)

    def relu_derivative(self, z):
        """
        Derivative of ReLU: f'(z) = 1 if z > 0, else 0

        Parameters
        ----------
        z : ndarray — the pre-activation values (cached during forward pass)

        Returns
        -------
        ndarray of same shape, dtype float64
        """
        return (z > 0).astype(np.float64)

    def sigmoid(self, z):
        """
        Sigmoid activation: f(z) = 1 / (1 + exp(-z))

        Uses numerically stable form:
          if z >= 0: 1 / (1 + exp(-z))
          if z <  0: exp(z) / (1 + exp(z))

        Parameters
        ----------
        z : ndarray of shape (n_samples, 1)

        Returns
        -------
        ndarray, same shape, values in (0, 1)
        """
        return np.where(
            z >= 0,
            1.0 / (1.0 + np.exp(-z)),
            np.exp(z) / (1.0 + np.exp(z))
        )

    def sigmoid_derivative(self, z):
        """
        Derivative of sigmoid: f'(z) = sigmoid(z) * (1 - sigmoid(z))

        Parameters
        ----------
        z : ndarray — pre-activation values

        Returns
        -------
        ndarray, same shape
        """
        s = self.sigmoid(z)
        return s * (1.0 - s)

    # ------------------------------------------------------------------
    # Forward Pass
    # ------------------------------------------------------------------

    def forward(self, X):
        """
        Forward pass through all layers.

        Steps:
          Z1 = X  @ W1 + b1  ;  A1 = relu(Z1)
          Z2 = A1 @ W2 + b2  ;  A2 = relu(Z2)
          Z3 = A2 @ W3 + b3  ;  A3 = sigmoid(Z3)  <- output

        All intermediate values are cached in self.cache for backprop.

        Parameters
        ----------
        X : ndarray of shape (n_samples, 30)

        Returns
        -------
        A3 : ndarray of shape (n_samples, 1) — predicted probabilities
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # Layer 1
        Z1 = X @ W1 + b1          # (n, 16)
        A1 = self.relu(Z1)        # (n, 16)

        # Layer 2
        Z2 = A1 @ W2 + b2         # (n, 8)
        A2 = self.relu(Z2)        # (n, 8)

        # Layer 3 (output)
        Z3 = A2 @ W3 + b3         # (n, 1)
        A3 = self.sigmoid(Z3)     # (n, 1)

        # Cache everything needed for backprop
        self.cache = {
            'Z1': Z1, 'A1': A1,
            'Z2': Z2, 'A2': A2,
            'Z3': Z3, 'A3': A3,
        }

        return A3

    # ------------------------------------------------------------------
    # Loss Function
    # ------------------------------------------------------------------

    def binary_cross_entropy(self, y_true, y_pred):
        """
        Numerically stable Binary Cross-Entropy loss:
          L = -mean( y * log(p) + (1-y) * log(1-p) )

        Clipping p to [eps, 1-eps] prevents log(0) = -inf.

        Parameters
        ----------
        y_true : ndarray of shape (n, 1) — ground truth labels {0, 1}
        y_pred : ndarray of shape (n, 1) — predicted probabilities

        Returns
        -------
        float — scalar loss value
        """
        eps = 1e-12
        p = np.clip(y_pred, eps, 1.0 - eps)
        loss = -np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))
        return float(loss)

    # ------------------------------------------------------------------
    # Backward Pass (Manual Backpropagation)
    # ------------------------------------------------------------------

    def backward(self, X, y_true):
        """
        Manual backpropagation using the chain rule.

        Derivations
        -----------
        n = number of samples

        Output layer (BCE + Sigmoid combined):
          dL/dZ3 = (A3 - y) / n       <- neat simplification

        Layer 3 gradients:
          dW3 = A2.T @ dZ3
          db3 = sum(dZ3, axis=0)
          dA2 = dZ3 @ W3.T

        Layer 2 (ReLU):
          dZ2 = dA2 * relu'(Z2)
          dW2 = A1.T @ dZ2
          db2 = sum(dZ2, axis=0)
          dA1 = dZ2 @ W2.T

        Layer 1 (ReLU):
          dZ1 = dA1 * relu'(Z1)
          dW1 = X.T  @ dZ1
          db1 = sum(dZ1, axis=0)

        Parameters
        ----------
        X      : ndarray (n, 30) — input data
        y_true : ndarray (n, 1)  — true labels

        Returns
        -------
        grads : dict with keys W1,b1,W2,b2,W3,b3
        """
        n = X.shape[0]

        # Retrieve cached values from forward pass
        Z1, A1 = self.cache['Z1'], self.cache['A1']
        Z2, A2 = self.cache['Z2'], self.cache['A2']
        Z3, A3 = self.cache['Z3'], self.cache['A3']

        W2, W3 = self.params['W2'], self.params['W3']

        # ---- Layer 3 ----
        # Combined gradient of BCE loss + sigmoid activation:
        # dL/dZ3 = (A3 - y) / n
        dZ3 = (A3 - y_true) / n          # (n, 1)
        dW3 = A2.T @ dZ3                  # (8, 1)
        db3 = np.sum(dZ3, axis=0, keepdims=True)  # (1, 1)
        dA2 = dZ3 @ W3.T                  # (n, 8)

        # ---- Layer 2 (ReLU) ----
        # dL/dZ2 = dL/dA2 * dA2/dZ2
        dZ2 = dA2 * self.relu_derivative(Z2)  # (n, 8)
        dW2 = A1.T @ dZ2                       # (16, 8)
        db2 = np.sum(dZ2, axis=0, keepdims=True)  # (1, 8)
        dA1 = dZ2 @ W2.T                       # (n, 16)

        # ---- Layer 1 (ReLU) ----
        dZ1 = dA1 * self.relu_derivative(Z1)  # (n, 16)
        dW1 = X.T @ dZ1                        # (30, 16)
        db1 = np.sum(dZ1, axis=0, keepdims=True)  # (1, 16)

        grads = {
            'W1': dW1, 'b1': db1,
            'W2': dW2, 'b2': db2,
            'W3': dW3, 'b3': db3,
        }
        return grads

    # ------------------------------------------------------------------
    # Weight Update
    # ------------------------------------------------------------------

    def update_weights(self, grads, lr):
        """
        Vanilla gradient descent update:
          W = W - lr * dW
          b = b - lr * db

        Parameters
        ----------
        grads : dict — output of backward()
        lr    : float — learning rate
        """
        num_layers = len(self.layer_sizes) - 1
        for l in range(1, num_layers + 1):
            self.params[f'W{l}'] -= lr * grads[f'W{l}']
            self.params[f'b{l}'] -= lr * grads[f'b{l}']

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X, threshold=0.5):
        """
        Convert predicted probabilities to binary class labels.

        Parameters
        ----------
        X         : ndarray (n, 30)
        threshold : float — decision boundary (default 0.5)

        Returns
        -------
        ndarray of shape (n, 1), dtype int — predicted labels {0, 1}
        """
        probs = self.forward(X)
        return (probs >= threshold).astype(np.int32)

    # ------------------------------------------------------------------
    # Parameter Serialization
    # ------------------------------------------------------------------

    def get_params(self):
        """
        Return a copy of all model parameters.

        Returns
        -------
        dict — {W1, b1, W2, b2, W3, b3} as numpy arrays
        """
        return {k: v.copy() for k, v in self.params.items()}

    def set_params(self, params):
        """
        Load parameters into the model (e.g., from a saved file or PyTorch).

        Parameters
        ----------
        params : dict — must contain W1,b1,W2,b2,W3,b3 as numpy arrays
        """
        for k, v in params.items():
            self.params[k] = np.array(v, dtype=np.float64)
