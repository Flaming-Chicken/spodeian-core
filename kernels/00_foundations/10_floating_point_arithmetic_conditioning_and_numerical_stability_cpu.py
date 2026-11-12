"""Pure-CPU tiled/vectorized hardware kernels for Topic 10:
10_floating_point_arithmetic_conditioning_and_numerical_stability
"""

from __future__ import annotations

import numpy as np


def welford_variance_cpu(
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Single-pass numerically stable Welford online mean and variance kernel."""
    n, d = x.shape
    mean = np.zeros(d, dtype=np.float32)
    m2 = np.zeros(d, dtype=np.float32)

    for i in range(n):
        xi = x[i]
        delta = xi - mean
        mean += delta / float(i + 1)
        delta2 = xi - mean
        m2 += delta * delta2

    var = m2 / float(n)
    return mean, var


def log_sum_exp_cpu(
    x: np.ndarray,
    axis: int = -1,
) -> np.ndarray:
    """Tiled Log-Sum-Exp kernel with max subtraction to prevent overflow/underflow."""
    m = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(x - m), axis=axis))


def neumaier_sum_cpu(
    x: np.ndarray,
    axis: int = 0,
) -> np.ndarray:
    """Neumaier compensated summation kernel recovering lost rounding bits."""
    n = x.shape[axis]
    d = x.shape[1]
    s = np.zeros(d, dtype=np.float32)
    c = np.zeros(d, dtype=np.float32)

    for i in range(n):
        xi = x[i]
        t = s + xi
        mask = np.abs(s) >= np.abs(xi)
        comp = np.where(mask, (s - t) + xi, (xi - t) + s)
        c += comp
        s = t

    return s + c


def matrix_condition_number_cpu(
    a: np.ndarray,
) -> float:
    """Computes matrix 2-norm condition number via SVD singular values."""
    s = np.linalg.svd(a, compute_uv=False)
    s_max = float(s[0])
    s_min = float(s[-1])
    if s_min == 0.0:
        return float("inf")
    return s_max / s_min


def stable_linear_gemm_cpu(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """Direct BLAS GEMM for linear projection: x @ weight + bias."""
    return np.matmul(x, weight) + bias
