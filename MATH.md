# Mathematical Foundations of the Neural Network

This document provides a thorough mathematical treatment of the neural network implemented from scratch in this project. All derivations are presented using plain ASCII math notation for maximum compatibility.

---

## 1. Network Architecture Overview

The network is a fully-connected (dense) feedforward neural network with two hidden layers.

```
Input Layer        Hidden Layer 1     Hidden Layer 2     Output Layer
  (30 units)         (16 units)          (8 units)         (1 unit)

 [x_1  ]          [h1_1 ]            [h2_1 ]
 [x_2  ]          [h1_2 ]            [h2_2]
 [ ... ] --FC-->  [ ... ] --ReLU-->  [ ... ] --ReLU-->  [z3] --Sigmoid--> [y_hat]
 [x_30 ]          [h1_16]            [h2_8 ]
```

**Compact Diagram:**

```
Input(30) --> FC(30x16) --> ReLU --> FC(16x8) --> ReLU --> FC(8x1) --> Sigmoid --> Output(1)
```

| Layer    | Operation           | Input Dim | Output Dim | Activation |
|----------|---------------------|-----------|------------|------------|
| Layer 1  | Z1 = X·W1 + b1      | (m, 30)   | (m, 16)    | ReLU       |
| Layer 2  | Z2 = A1·W2 + b2     | (m, 16)   | (m, 8)     | ReLU       |
| Layer 3  | Z3 = A2·W3 + b3     | (m, 8)    | (m, 1)     | Sigmoid    |

Where `m` is the number of training samples (batch size).

---

## 2. Forward Propagation

### Weight Initialization (He Initialization)

For ReLU-activated layers, He initialization is used to prevent vanishing/exploding gradients:

```
W ~ Normal(0, 1) * sqrt(2 / fan_in)
b = zeros
```

- W1: shape (30, 16)
- b1: shape (1, 16)
- W2: shape (16, 8)
- b2: shape (1, 8)
- W3: shape (8, 1)
- b3: shape (1, 1)

### Layer 1 — First Hidden Layer

```
Z1 = X · W1 + b1
A1 = ReLU(Z1)
```

Dimensions:
```
X:  (m, 30)
W1: (30, 16)
b1: (1,  16)   <- broadcast across m rows
Z1: (m, 16)
A1: (m, 16)
```

### Layer 2 — Second Hidden Layer

```
Z2 = A1 · W2 + b2
A2 = ReLU(Z2)
```

Dimensions:
```
A1: (m, 16)
W2: (16, 8)
b2: (1,  8)   <- broadcast
Z2: (m, 8)
A2: (m, 8)
```

### Layer 3 — Output Layer

```
Z3 = A2 · W3 + b3
A3 = sigmoid(Z3)
```

Dimensions:
```
A2: (m, 8)
W3: (8, 1)
b3: (1, 1)   <- broadcast
Z3: (m, 1)
A3: (m, 1)   <- predicted probabilities y_hat
```

---

## 3. Activation Functions

### ReLU (Rectified Linear Unit)

```
ReLU(z) = max(0, z)
```

Applied element-wise. Clips all negative values to zero.

**Derivative (for backpropagation):**

```
ReLU'(z) = 1  if z > 0
           0  if z <= 0
```

This is a step function (sub-gradient at z=0 is taken as 0).

**Why ReLU?**
- Computationally cheap (simple thresholding)
- Does not saturate for positive values (avoids vanishing gradients)
- Empirically works well in hidden layers

### Sigmoid

```
sigmoid(z) = 1 / (1 + exp(-z))
```

Squashes output to the range (0, 1), making it interpretable as a probability.

**Derivative:**

```
sigmoid'(z) = sigma(z) * (1 - sigma(z))
```

**Derivation:**

```
Let s = 1 / (1 + e^{-z})

ds/dz = e^{-z} / (1 + e^{-z})^2
      = [1/(1+e^{-z})] * [e^{-z}/(1+e^{-z})]
      = s * (1 - s)
```

**Why Sigmoid at output?**
- Output is in (0, 1) — natural probability interpretation
- Pairs cleanly with Binary Cross-Entropy loss (gradients simplify nicely)

---

## 4. Loss Function

### Binary Cross-Entropy (BCE)

For binary classification, the negative log-likelihood under a Bernoulli distribution gives:

```
L = -(1/m) * sum_i [ y_i * log(y_hat_i) + (1 - y_i) * log(1 - y_hat_i) ]
```

Where:
- `m`     = number of samples
- `y_i`   = true label for sample i (0 or 1)
- `y_hat_i` = predicted probability for sample i

**Interpretation:**
- When y=1: loss = -log(y_hat) — penalizes low confidence in correct class
- When y=0: loss = -log(1-y_hat) — penalizes high confidence in wrong class

### Numerical Stability

The logarithm is undefined at 0 and negative. To prevent `log(0) = -inf`, predictions are clipped:

```
y_hat_clipped = clip(y_hat, epsilon, 1 - epsilon)
              where epsilon = 1e-15
```

This ensures that `y_hat` is always strictly in (1e-15, 1-1e-15) before taking the log.

---

## 5. Backpropagation

Backpropagation applies the **chain rule** of calculus to compute the gradient of the loss L with respect to every trainable parameter (W1, b1, W2, b2, W3, b3).

### Chain Rule Principle

For a composition of functions:
```
L = f(g(h(x)))

dL/dx = (dL/df) * (df/dg) * (dg/dh) * (dh/dx)
```

We propagate gradients backwards from the output layer to the input.

---

### Layer 3 — Output Layer Gradients

#### Combined BCE + Sigmoid Gradient

The combined gradient `dL/dZ3` simplifies beautifully when BCE loss is used with Sigmoid:

```
dL/dZ3 = dL/dA3 * dA3/dZ3
```

Starting from:
```
dL/dA3 = -(y/A3) + (1-y)/(1-A3)
dA3/dZ3 = A3 * (1 - A3)   [sigmoid derivative]

dL/dZ3 = [-(y/A3) + (1-y)/(1-A3)] * A3*(1-A3)
        = -y*(1-A3) + (1-y)*A3
        = A3 - y
```

So:
```
dZ3 = A3 - y         shape: (m, 1)
```

This is the elegant result: the gradient at the output pre-activation is simply the residual (prediction minus truth).

#### Gradients for W3, b3

```
dW3 = (1/m) * A2^T · dZ3        shape: (8, 1)
db3 = (1/m) * sum(dZ3, axis=0)  shape: (1, 1)
```

**Derivation of dW3:**
```
Z3 = A2 · W3 + b3
dL/dW3 = A2^T · dL/dZ3  (averaged over m)
```

---

### Layer 2 — Second Hidden Layer Gradients

First, propagate gradient back through W3 to A2:
```
dA2 = dZ3 · W3^T          shape: (m, 8)
```

Then apply the ReLU derivative (element-wise Hadamard product, denoted ⊙):
```
dZ2 = dA2 ⊙ ReLU'(Z2)    shape: (m, 8)
```

Where `ReLU'(Z2)` is 1 where Z2 > 0, else 0.

Then compute weight gradients:
```
dW2 = (1/m) * A1^T · dZ2        shape: (16, 8)
db2 = (1/m) * sum(dZ2, axis=0)  shape: (1,  8)
```

---

### Layer 1 — First Hidden Layer Gradients

Propagate back through W2 to A1:
```
dA1 = dZ2 · W2^T          shape: (m, 16)
```

Apply ReLU derivative:
```
dZ1 = dA1 ⊙ ReLU'(Z1)    shape: (m, 16)
```

Compute weight gradients:
```
dW1 = (1/m) * X^T · dZ1         shape: (30, 16)
db1 = (1/m) * sum(dZ1, axis=0)  shape: (1,  16)
```

---

### Summary Table of Backpropagation

| Step | Equation                                     | Shape of Result |
|------|----------------------------------------------|-----------------|
| dZ3  | A3 - y                                       | (m, 1)          |
| dW3  | (1/m) * A2^T · dZ3                           | (8, 1)          |
| db3  | (1/m) * sum(dZ3, axis=0)                     | (1, 1)          |
| dA2  | dZ3 · W3^T                                   | (m, 8)          |
| dZ2  | dA2 ⊙ ReLU'(Z2)                              | (m, 8)          |
| dW2  | (1/m) * A1^T · dZ2                           | (16, 8)         |
| db2  | (1/m) * sum(dZ2, axis=0)                     | (1, 8)          |
| dA1  | dZ2 · W2^T                                   | (m, 16)         |
| dZ1  | dA1 ⊙ ReLU'(Z1)                              | (m, 16)         |
| dW1  | (1/m) * X^T · dZ1                            | (30, 16)        |
| db1  | (1/m) * sum(dZ1, axis=0)                     | (1, 16)         |

---

## 6. Weight Updates (Gradient Descent)

After computing all gradients, each parameter is updated by stepping in the negative gradient direction:

```
W := W - lr * dW
b := b - lr * db
```

Where:
- `lr` = learning rate (0.01 in this project)
- `dW`, `db` = gradients of loss with respect to that parameter

**Full-batch gradient descent** is used: all training samples are used in each update step, providing a stable (but slow) gradient estimate.

---

## 7. Numerical Gradient Checking

Gradient checking is a debugging technique that verifies the analytical (backprop) gradients against numerically estimated gradients.

### Finite Difference Approximation

By definition of the derivative:

```
f'(theta) = lim_{eps->0} [f(theta + eps) - f(theta - eps)] / (2*eps)
```

For a small but finite epsilon (eps = 1e-5), this **centered finite difference** provides an accurate numerical estimate:

```
grad_numerical(theta_i) = [L(theta_i + eps) - L(theta_i - eps)] / (2 * eps)
```

For each scalar parameter `theta_i`, we:
1. Set `theta_i += eps`, compute loss L_plus
2. Set `theta_i -= eps` (restore), set `theta_i -= eps`, compute L_minus
3. Restore `theta_i` to original value
4. Compute `grad_numerical = (L_plus - L_minus) / (2 * eps)`

### Relative Error

To compare the analytical gradient `grad_analytical` to the numerical gradient `grad_numerical`:

```
relative_error = |grad_analytical - grad_numerical|
                 ----------------------------------------
                 max(|grad_analytical|, |grad_numerical|) + eps
```

The `eps` in the denominator prevents division by zero when both gradients are near zero.

### Tolerance and Interpretation

| Relative Error | Interpretation                                      |
|---------------|------------------------------------------------------|
| < 1e-7        | Excellent — almost certainly correct                 |
| < 1e-5        | Good — likely correct (our PASS threshold)           |
| < 1e-3        | Suspicious — worth investigating                     |
| >= 1e-3       | FAIL — likely a bug in the backpropagation           |

**Tolerance used in this project: relative error < 1e-5 is considered PASS.**

The gradient check is run on a small deterministic subset (5 samples, seed=42) for efficiency — checking all 455 training samples would be prohibitively slow since each scalar parameter requires two full forward passes.

> **See `results/gradient_check.txt` for the actual gradient check output and per-parameter pass/fail results.**

---

## Appendix: Notation Reference

| Symbol       | Meaning                                      |
|--------------|----------------------------------------------|
| m            | Number of training samples (batch size)      |
| n_x          | Number of input features (30)                |
| X            | Input matrix, shape (m, n_x)                 |
| y            | True labels, shape (m, 1)                    |
| y_hat / A3   | Predicted probabilities, shape (m, 1)        |
| W1, W2, W3   | Weight matrices for layers 1, 2, 3           |
| b1, b2, b3   | Bias vectors for layers 1, 2, 3              |
| Z1, Z2, Z3   | Pre-activation values                        |
| A1, A2, A3   | Post-activation values                       |
| L            | Scalar loss value                            |
| lr           | Learning rate (0.01)                         |
| eps / ε      | Small constant (1e-5 for gradient check)     |
| ⊙            | Element-wise (Hadamard) product              |
| ^T           | Matrix transpose                             |
