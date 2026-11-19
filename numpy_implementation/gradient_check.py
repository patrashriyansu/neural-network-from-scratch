"""
Numerical Gradient Checking for the NumPy Neural Network.

Verifies that the analytical (backprop) gradients match numerical gradients
computed via finite differences. This is a standard sanity check for
hand-coded backpropagation.

Method (centered finite differences):
  dL/dθ ≈ [L(θ + ε) - L(θ - ε)] / (2ε)

Comparison metric (relative error):
  rel_err = |grad_analytical - grad_numerical| /
            (|grad_analytical| + |grad_numerical| + tiny)

A relative error < 1e-5 is considered PASS.
"""

import os
import sys
import numpy as np

# Allow running as a script from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from numpy_implementation.model import NeuralNetwork


def _compute_loss(model, X, y):
    """Helper: run a forward pass and return the scalar BCE loss."""
    y_pred = model.forward(X)
    return model.binary_cross_entropy(y, y_pred)


def check_gradients(model, X_small, y_small, epsilon=1e-5):
    """
    Numerical gradient check using centered finite differences.

    For every trainable parameter (W1, b1, W2, b2, W3, b3):
      1. Get the analytical gradient via forward + backward pass.
      2. For each scalar element θ_i:
           L_plus  = loss(θ_i + ε)
           L_minus = loss(θ_i - ε)
           grad_num_i = (L_plus - L_minus) / (2ε)
      3. Compare analytical vs numerical gradient:
           abs_err = |grad_analytical - grad_numerical|
           rel_err = abs_err / (|grad_analytical| + |grad_numerical| + 1e-15)

    Parameters
    ----------
    model   : NeuralNetwork — model to check (freshly initialized)
    X_small : ndarray (n, 30)
    y_small : ndarray (n, 1)
    epsilon : float — perturbation size (default 1e-5)

    Returns
    -------
    report : list of dicts — one dict per parameter with keys:
               name, max_abs_err, max_rel_err, passed
    """
    # ---- Step 1: Analytical gradients ----
    model.forward(X_small)                        # populate cache
    grads_analytical = model.backward(X_small, y_small)

    report = []
    param_names = ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']

    for name in param_names:
        param = model.params[name]                # reference to param array
        grad_analytic = grads_analytical[name]    # same shape

        grad_numerical = np.zeros_like(param)
        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])

        while not it.finished:
            idx = it.multi_index
            original = float(param[idx])

            # L(θ + ε)
            param[idx] = original + epsilon
            loss_plus = _compute_loss(model, X_small, y_small)

            # L(θ - ε)
            param[idx] = original - epsilon
            loss_minus = _compute_loss(model, X_small, y_small)

            # Restore
            param[idx] = original

            # Centered finite difference
            grad_numerical[idx] = (loss_plus - loss_minus) / (2.0 * epsilon)
            it.iternext()

        # Compute errors
        abs_err = np.abs(grad_analytic - grad_numerical)
        denom   = np.abs(grad_analytic) + np.abs(grad_numerical) + 1e-15
        rel_err = abs_err / denom

        max_abs = float(np.max(abs_err))
        max_rel = float(np.max(rel_err))
        passed  = max_rel < 1e-5

        report.append({
            'name':        name,
            'shape':       param.shape,
            'max_abs_err': max_abs,
            'max_rel_err': max_rel,
            'passed':      passed,
        })

    return report


def run_gradient_check():
    """
    Run gradient check on small deterministic synthetic data and save report.
    """
    # ---- Deterministic small dataset (5 samples, 30 features) ----
    rng = np.random.default_rng(42)
    n_samples = 5
    n_features = 30
    X_small = rng.standard_normal((n_samples, n_features)).astype(np.float64)
    y_small = rng.integers(0, 2, size=(n_samples, 1)).astype(np.float64)

    # Fresh model
    model = NeuralNetwork(layer_sizes=[30, 16, 8, 1], seed=42)

    print("Running numerical gradient check...")
    print(f"  Data shape: X={X_small.shape}, y={y_small.shape}")
    print(f"  Epsilon: 1e-5  |  Tolerance (rel error): 1e-5\n")

    report = check_gradients(model, X_small, y_small)

    # ---- Print report ----
    header = f"{'Param':<6}  {'Shape':<12}  {'Max Abs Err':>14}  {'Max Rel Err':>14}  {'Result'}"
    separator = "-" * len(header)
    print(header)
    print(separator)

    all_passed = True
    lines = [header, separator]

    for r in report:
        shape_str = str(r['shape'])
        result = "PASS ✓" if r['passed'] else "FAIL ✗"
        if not r['passed']:
            all_passed = False
        row = (f"{r['name']:<6}  {shape_str:<12}  "
               f"{r['max_abs_err']:>14.2e}  {r['max_rel_err']:>14.2e}  {result}")
        print(row)
        lines.append(row)

    print(separator)
    final_verdict = "ALL CHECKS PASSED ✓" if all_passed else "SOME CHECKS FAILED ✗"
    print(f"\n{final_verdict}\n")
    lines += [separator, "", final_verdict, ""]

    # ---- Save report ----
    os.makedirs('results', exist_ok=True)
    report_path = 'results/gradient_check.txt'
    with open(report_path, 'w') as f:
        f.write("Numerical Gradient Check Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Architecture : [30, 16, 8, 1]\n")
        f.write(f"Dataset      : 5 samples, 30 features (synthetic, seed=42)\n")
        f.write(f"Epsilon      : 1e-5\n")
        f.write(f"Tolerance    : relative error < 1e-5 -> PASS\n")
        f.write("=" * 60 + "\n\n")
        for line in lines:
            f.write(line + "\n")

        f.write("\nDetailed per-parameter stats:\n")
        for r in report:
            f.write(f"\n{r['name']} {r['shape']}:\n")
            f.write(f"  Max absolute error : {r['max_abs_err']:.4e}\n")
            f.write(f"  Max relative error : {r['max_rel_err']:.4e}\n")
            f.write(f"  Result             : {'PASS' if r['passed'] else 'FAIL'}\n")

    print(f"Full report saved to: {report_path}")
    return report, all_passed


if __name__ == "__main__":
    report, all_passed = run_gradient_check()
    sys.exit(0 if all_passed else 1)
