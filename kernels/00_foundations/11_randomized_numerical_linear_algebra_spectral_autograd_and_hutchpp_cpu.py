"""Pure-CPU vectorized hardware kernels for Topic 11:
11_randomized_numerical_linear_algebra_spectral_autograd_and_hutchpp
"""

from __future__ import annotations

import numpy as np


def modified_gram_schmidt_qr_cpu(
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pure-CPU vectorized Modified Gram-Schmidt thin QR decomposition."""
    m, num_cols = y.shape
    q = y.astype(np.float32, copy=True)
    r = np.zeros((num_cols, num_cols), dtype=np.float32)

    for j in range(num_cols):
        v = q[:, j].copy()
        for i in range(j):
            rij = float(np.dot(q[:, i], v))
            r[i, j] = rij
            v -= rij * q[:, i]
        rjj = float(np.linalg.norm(v))
        r[j, j] = rjj
        q[:, j] = v / (rjj + 1e-12)

    return q, r


def randomized_range_finder_cpu(
    a: np.ndarray,
    omega: np.ndarray,
) -> np.ndarray:
    """CPU kernel forming orthonormal sketch Q spanning dominant range of A."""
    y = np.dot(a, omega)
    q, _ = modified_gram_schmidt_qr_cpu(y)
    return q


def hutch_plus_plus_trace_cpu(
    a: np.ndarray,
    s_sketch: np.ndarray,
    g_mc: np.ndarray,
) -> np.ndarray:
    """CPU kernel for Hutch++ randomized trace estimator with O(1/eps) query complexity."""
    # 1. Low-rank subspace sketch
    y = np.dot(a, s_sketch)
    q, _ = modified_gram_schmidt_qr_cpu(y)

    # 2. Exact trace of projection on Q: tr(Q^T A Q)
    q_t_a_q = np.dot(np.dot(q.T, a), q)
    tr_proj = float(np.trace(q_t_a_q))

    # 3. Residual Monte Carlo on (I - Q Q^T) A
    a_g = np.dot(a, g_mc)
    proj_g = np.dot(q, np.dot(q.T, a_g))
    residual_g = a_g - proj_g

    m_val = float(g_mc.shape[1])
    quad_forms = np.sum(g_mc * residual_g, axis=0)
    tr_res = float(np.sum(quad_forms) / m_val)

    return np.array([tr_proj + tr_res], dtype=np.float32)


def lorentzian_broadened_f_matrix_cpu(
    singular_values: np.ndarray,
    epsilon: float = 1e-3,
) -> np.ndarray:
    """CPU kernel evaluating skew-symmetric Lorentzian-broadened singular value coupling matrix."""
    s = np.asarray(singular_values, dtype=np.float32)
    n_dim = s.shape[0]
    delta = s[np.newaxis, :] - s[:, np.newaxis]

    f = delta / (delta**2 + (epsilon**2))
    f *= 1.0 - np.eye(n_dim, dtype=np.float32)
    return f


def power_iteration_spectral_norm_cpu(
    w: np.ndarray,
    v_init: np.ndarray,
    num_iters: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CPU kernel evaluating dominant singular value and vectors via Power Iteration."""
    w_mat = np.asarray(w, dtype=np.float32)
    v = np.asarray(v_init, dtype=np.float32)
    v /= float(np.linalg.norm(v)) + 1e-12
    u = np.zeros(w_mat.shape[0], dtype=np.float32)

    for _ in range(num_iters):
        u_unnorm = np.dot(w_mat, v)
        u = u_unnorm / (float(np.linalg.norm(u_unnorm)) + 1e-12)
        v_unnorm = np.dot(w_mat.T, u)
        v = v_unnorm / (float(np.linalg.norm(v_unnorm)) + 1e-12)

    sigma = float(np.dot(u, np.dot(w_mat, v)))
    return np.array([sigma], dtype=np.float32), u, v
