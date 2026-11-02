"""Pure-CPU vectorized hardware kernels for Category 01 Topic 06:
06_ai_by_purpose_prediction_prescription_control_combinatorial_symbolic
"""

from __future__ import annotations

import numpy as np


def cost_predictor_forward_cpu(
    x: np.ndarray,
    w_cost1: np.ndarray,
    b_cost1: np.ndarray,
    w_cost2: np.ndarray,
    b_cost2: np.ndarray,
) -> np.ndarray:
    """Vectorized CPU kernel for cost prediction: softplus(ReLU(x @ w1 + b1) @ w2 + b2)."""
    x_arr = np.asarray(x, dtype=np.float32)
    w1 = np.asarray(w_cost1, dtype=np.float32)
    b1 = np.asarray(b_cost1, dtype=np.float32)
    w2 = np.asarray(w_cost2, dtype=np.float32)
    b2 = np.asarray(b_cost2, dtype=np.float32)

    h = np.maximum(np.dot(x_arr, w1) + b1, 0.0)
    out = np.dot(h, w2) + b2
    # Numerically stable softplus: log(1 + exp(out)) via logaddexp
    return np.logaddexp(out, 0.0)


def solve_lp_simplex_cpu(cost_vectors: np.ndarray) -> np.ndarray:
    """Vectorized CPU kernel solving exact simplex LP: one_hot(argmin_i c_i)."""
    c_arr = np.asarray(cost_vectors, dtype=np.float32)
    opt_idx = np.argmin(c_arr, axis=-1)
    cost_dim = c_arr.shape[-1]
    return (opt_idx[..., None] == np.arange(cost_dim)).astype(np.float32)


def decision_regret_cpu(c_pred: np.ndarray, c_star: np.ndarray) -> np.ndarray:
    """Vectorized CPU kernel computing decision regret: c*^T w*(c_pred) - c*^T w*(c*)."""
    cp = np.asarray(c_pred, dtype=np.float32)
    cs = np.asarray(c_star, dtype=np.float32)

    w_pred = solve_lp_simplex_cpu(cp)
    w_star = solve_lp_simplex_cpu(cs)

    cost_actual = np.sum(cs * w_pred, axis=-1)
    cost_oracle = np.sum(cs * w_star, axis=-1)
    return cost_actual - cost_oracle


def spo_plus_loss_cpu(c_pred: np.ndarray, c_star: np.ndarray) -> np.ndarray:
    """Vectorized CPU kernel computing convex SPO+ loss."""
    cp = np.asarray(c_pred, dtype=np.float32)
    cs = np.asarray(c_star, dtype=np.float32)

    w_star = solve_lp_simplex_cpu(cs)
    cost_oracle = np.sum(cs * w_star, axis=-1)

    v_vec = cs - 2.0 * cp
    max_term = np.max(v_vec, axis=-1)
    lin_term = 2.0 * np.sum(cp * w_star, axis=-1) - cost_oracle
    return max_term + lin_term


def causal_forward_cpu(
    x: np.ndarray,
    treatment_t: np.ndarray,
    w_tau: np.ndarray,
    b_tau: np.ndarray,
    w_mu0: np.ndarray,
    b_mu0: np.ndarray,
) -> np.ndarray:
    """Vectorized CPU kernel for causal potential outcomes: mu0(x) + T * tau(x)."""
    x_arr = np.asarray(x, dtype=np.float32)
    t_arr = np.asarray(treatment_t, dtype=np.float32)
    wt = np.asarray(w_tau, dtype=np.float32)
    bt = np.asarray(b_tau, dtype=np.float32)
    wm = np.asarray(w_mu0, dtype=np.float32)
    bm = np.asarray(b_mu0, dtype=np.float32)

    tau = np.dot(x_arr, wt) + bt
    mu0 = np.dot(x_arr, wm) + bm
    return mu0 + t_arr * tau


def symbolic_forward_cpu(x: np.ndarray, w_eql: np.ndarray, b_eql: np.ndarray) -> np.ndarray:
    """Vectorized CPU kernel for EQL symbolic primitives."""
    x_arr = np.asarray(x, dtype=np.float32)
    w = np.asarray(w_eql, dtype=np.float32)
    b = np.asarray(b_eql, dtype=np.float32)

    f0 = x_arr[:, 0:1]
    f1 = np.sin(x_arr[:, 1:2])
    f2 = np.cos(x_arr[:, 2:3])
    f3 = x_arr[:, 0:1] * x_arr[:, 3:4]
    features = np.concatenate([f0, f1, f2, f3], axis=-1)
    return np.dot(features, w) + b
