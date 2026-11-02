"""Pure-CPU vectorized hardware kernels for Category 01 Topic 05:
05_parametric_vs_nonparametric_and_neural_processes
"""

from __future__ import annotations

import numpy as np


def parametric_forward_cpu(
    x_target: np.ndarray,
    w_mlp1: np.ndarray,
    b_mlp1: np.ndarray,
    w_mlp2: np.ndarray,
    b_mlp2: np.ndarray,
) -> np.ndarray:
    """Vectorized CPU kernel for parametric MLP."""
    xt = np.asarray(x_target, dtype=np.float32)
    w1 = np.asarray(w_mlp1, dtype=np.float32)
    b1 = np.asarray(b_mlp1, dtype=np.float32)
    w2 = np.asarray(w_mlp2, dtype=np.float32)
    b2 = np.asarray(b_mlp2, dtype=np.float32)

    h = np.maximum(np.dot(xt, w1) + b1, 0.0)
    return np.dot(h, w2) + b2


def gp_rbf_kernel_cpu(x1: np.ndarray, x2: np.ndarray, lengthscale: float | np.ndarray) -> np.ndarray:
    """Vectorized CPU kernel computing pairwise RBF covariance."""
    x1_arr = np.asarray(x1, dtype=np.float32)
    x2_arr = np.asarray(x2, dtype=np.float32)
    ls = np.asarray(lengthscale, dtype=np.float32)

    x1_sq = np.sum(np.square(x1_arr), axis=-1, keepdims=True)
    x2_sq = np.sum(np.square(x2_arr), axis=-1, keepdims=True)
    dist_sq = x1_sq + np.swapaxes(x2_sq, -2, -1) - 2.0 * np.matmul(x1_arr, np.swapaxes(x2_arr, -2, -1))
    dist_sq = np.maximum(dist_sq, 0.0)
    return np.exp(-0.5 * dist_sq / np.square(ls))


def gp_forward_cpu(
    x_context: np.ndarray,
    y_context: np.ndarray,
    x_target: np.ndarray,
    lengthscale: float | np.ndarray,
    noise_var: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized CPU kernel for Gaussian Process regularized solve."""
    xc = np.asarray(x_context, dtype=np.float32)
    yc = np.asarray(y_context, dtype=np.float32)
    xt = np.asarray(x_target, dtype=np.float32)
    ls = np.asarray(lengthscale, dtype=np.float32)
    nv = np.asarray(noise_var, dtype=np.float32)

    num_c = xc.shape[1]
    batch_size = xc.shape[0]

    k_cc = gp_rbf_kernel_cpu(xc, xc, ls)
    eye = np.broadcast_to(np.eye(num_c, dtype=np.float32), (batch_size, num_c, num_c))
    k_cc_reg = k_cc + nv * eye

    alpha = np.linalg.solve(k_cc_reg, yc)
    k_tc = gp_rbf_kernel_cpu(xt, xc, ls)
    y_pred = np.matmul(k_tc, alpha)
    return y_pred, alpha


def anp_forward_cpu(
    x_context: np.ndarray,
    y_context: np.ndarray,
    x_target: np.ndarray,
    w_enc: np.ndarray,
    b_enc: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_dec: np.ndarray,
    b_dec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized CPU kernel for Attentive Neural Process cross-attention."""
    xc = np.asarray(x_context, dtype=np.float32)
    yc = np.asarray(y_context, dtype=np.float32)
    xt = np.asarray(x_target, dtype=np.float32)

    we = np.asarray(w_enc, dtype=np.float32)
    be = np.asarray(b_enc, dtype=np.float32)
    wq = np.asarray(w_q, dtype=np.float32)
    wk = np.asarray(w_k, dtype=np.float32)
    wv = np.asarray(w_v, dtype=np.float32)
    wd = np.asarray(w_dec, dtype=np.float32)
    bd = np.asarray(b_dec, dtype=np.float32)

    xy_c = np.concatenate([xc, yc], axis=-1)
    v_c = np.maximum(np.dot(xy_c, we) + be, 0.0)

    q = np.dot(xt, wq)
    k = np.dot(xc, wk)
    v = np.dot(v_c, wv)

    scale = 1.0 / (float(q.shape[-1]) ** 0.5)
    scores = np.matmul(q, np.swapaxes(k, -2, -1)) * scale

    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    r_anp = np.matmul(attn_weights, v)
    xt_r = np.concatenate([xt, r_anp], axis=-1)
    y_pred = np.dot(xt_r, wd) + bd
    return y_pred, attn_weights


def cnp_forward_cpu(
    x_context: np.ndarray,
    y_context: np.ndarray,
    x_target: np.ndarray,
    w_enc: np.ndarray,
    b_enc: np.ndarray,
    w_dec: np.ndarray,
    b_dec: np.ndarray,
) -> np.ndarray:
    """Vectorized CPU kernel for Conditional Neural Process uniform mean pooling."""
    xc = np.asarray(x_context, dtype=np.float32)
    yc = np.asarray(y_context, dtype=np.float32)
    xt = np.asarray(x_target, dtype=np.float32)

    we = np.asarray(w_enc, dtype=np.float32)
    be = np.asarray(b_enc, dtype=np.float32)
    wd = np.asarray(w_dec, dtype=np.float32)
    bd = np.asarray(b_dec, dtype=np.float32)

    xy_c = np.concatenate([xc, yc], axis=-1)
    v_c = np.maximum(np.dot(xy_c, we) + be, 0.0)

    r_pooled = np.mean(v_c, axis=1, keepdims=True)
    num_target = xt.shape[1]
    r_expanded = np.broadcast_to(r_pooled, (xt.shape[0], num_target, r_pooled.shape[-1]))

    xt_r = np.concatenate([xt, r_expanded], axis=-1)
    return np.dot(xt_r, wd) + bd
