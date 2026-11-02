"""Pure-CPU vectorized hardware kernels for Category 01 Topic 04:
04_deterministic_vs_stochastic_and_mixture_density_networks
"""

from __future__ import annotations

import math

import numpy as np


def deterministic_forward_cpu(
    x: np.ndarray,
    w_h: np.ndarray,
    b_h: np.ndarray,
    w_det: np.ndarray,
    b_det: np.ndarray,
) -> np.ndarray:
    """Pure-CPU vectorized kernel for deterministic regression."""
    x_arr = np.asarray(x, dtype=np.float32)
    w_h_arr = np.asarray(w_h, dtype=np.float32)
    b_h_arr = np.asarray(b_h, dtype=np.float32)
    w_det_arr = np.asarray(w_det, dtype=np.float32)
    b_det_arr = np.asarray(b_det, dtype=np.float32)

    hidden = np.maximum(np.dot(x_arr, w_h_arr) + b_h_arr, 0.0)
    return np.dot(hidden, w_det_arr) + b_det_arr


def mdn_forward_cpu(
    x: np.ndarray,
    w_h: np.ndarray,
    b_h: np.ndarray,
    w_pi: np.ndarray,
    b_pi: np.ndarray,
    w_mu: np.ndarray,
    b_mu: np.ndarray,
    w_sigma: np.ndarray,
    b_sigma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure-CPU vectorized kernel for Mixture Density Network parameter extraction."""
    x_arr = np.asarray(x, dtype=np.float32)
    w_h_arr = np.asarray(w_h, dtype=np.float32)
    b_h_arr = np.asarray(b_h, dtype=np.float32)
    w_pi_arr = np.asarray(w_pi, dtype=np.float32)
    b_pi_arr = np.asarray(b_pi, dtype=np.float32)
    w_mu_arr = np.asarray(w_mu, dtype=np.float32)
    b_mu_arr = np.asarray(b_mu, dtype=np.float32)
    w_sigma_arr = np.asarray(w_sigma, dtype=np.float32)
    b_sigma_arr = np.asarray(b_sigma, dtype=np.float32)

    hidden = np.maximum(np.dot(x_arr, w_h_arr) + b_h_arr, 0.0)

    pi_logits = np.dot(hidden, w_pi_arr) + b_pi_arr
    max_logits = np.max(pi_logits, axis=-1, keepdims=True)
    exp_logits = np.exp(pi_logits - max_logits)
    pi = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    mu = np.dot(hidden, w_mu_arr) + b_mu_arr
    log_sigma = np.dot(hidden, w_sigma_arr) + b_sigma_arr
    sigma = np.exp(log_sigma)

    return pi, mu, sigma


def mdn_log_prob_cpu(
    x: np.ndarray,
    y: np.ndarray,
    w_h: np.ndarray,
    b_h: np.ndarray,
    w_pi: np.ndarray,
    b_pi: np.ndarray,
    w_mu: np.ndarray,
    b_mu: np.ndarray,
    w_sigma: np.ndarray,
    b_sigma: np.ndarray,
) -> np.ndarray:
    """Pure-CPU vectorized kernel computing Gaussian Mixture log-likelihood log p(y|x)."""
    x_arr = np.asarray(x, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    w_h_arr = np.asarray(w_h, dtype=np.float32)
    b_h_arr = np.asarray(b_h, dtype=np.float32)
    w_pi_arr = np.asarray(w_pi, dtype=np.float32)
    b_pi_arr = np.asarray(b_pi, dtype=np.float32)
    w_mu_arr = np.asarray(w_mu, dtype=np.float32)
    b_mu_arr = np.asarray(b_mu, dtype=np.float32)
    w_sigma_arr = np.asarray(w_sigma, dtype=np.float32)
    b_sigma_arr = np.asarray(b_sigma, dtype=np.float32)

    hidden = np.maximum(np.dot(x_arr, w_h_arr) + b_h_arr, 0.0)
    pi_logits = np.dot(hidden, w_pi_arr) + b_pi_arr

    max_logits = np.max(pi_logits, axis=-1, keepdims=True)
    log_pi = pi_logits - max_logits - np.log(np.sum(np.exp(pi_logits - max_logits), axis=-1, keepdims=True))

    mu = np.dot(hidden, w_mu_arr) + b_mu_arr
    log_sigma = np.dot(hidden, w_sigma_arr) + b_sigma_arr
    sigma = np.exp(log_sigma)

    diff = y_arr - mu
    log_gaussian = -0.5 * math.log(2.0 * math.pi) - log_sigma - 0.5 * np.square(diff / sigma)

    component_terms = log_pi + log_gaussian
    max_terms = np.max(component_terms, axis=-1, keepdims=True)
    return np.squeeze(max_terms, axis=-1) + np.log(np.sum(np.exp(component_terms - max_terms), axis=-1))


def quantile_forward_cpu(
    x: np.ndarray,
    w_h: np.ndarray,
    b_h: np.ndarray,
    w_quantile: np.ndarray,
    b_quantile: np.ndarray,
) -> np.ndarray:
    """Pure-CPU vectorized kernel for quantile predictions."""
    x_arr = np.asarray(x, dtype=np.float32)
    w_h_arr = np.asarray(w_h, dtype=np.float32)
    b_h_arr = np.asarray(b_h, dtype=np.float32)
    w_q_arr = np.asarray(w_quantile, dtype=np.float32)
    b_q_arr = np.asarray(b_quantile, dtype=np.float32)

    hidden = np.maximum(np.dot(x_arr, w_h_arr) + b_h_arr, 0.0)
    return np.dot(hidden, w_q_arr) + b_q_arr


def pinball_loss_cpu(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    quantiles: np.ndarray,
) -> np.ndarray:
    """Pure-CPU vectorized kernel for pinball loss."""
    yp_arr = np.asarray(y_pred, dtype=np.float32)
    yt_arr = np.asarray(y_true, dtype=np.float32)
    q_arr = np.asarray(quantiles, dtype=np.float32)

    u = yt_arr - yp_arr
    return np.maximum(q_arr * u, (q_arr - 1.0) * u)
