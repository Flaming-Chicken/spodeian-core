"""Pure-CPU vectorized hardware kernels for Category 01 Topic 03:
03_discrete_vs_continuous_latents_and_gradient_estimators
"""

from __future__ import annotations

import numpy as np


def gaussian_encode_cpu(
    x: np.ndarray,
    w_mu: np.ndarray,
    b_mu: np.ndarray,
    w_logvar: np.ndarray,
    b_logvar: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized CPU kernel evaluating Gaussian encoder mean and log-variance."""
    x_arr = np.asarray(x, dtype=np.float32)
    w_m = np.asarray(w_mu, dtype=np.float32)
    b_m = np.asarray(b_mu, dtype=np.float32)
    w_lv = np.asarray(w_logvar, dtype=np.float32)
    b_lv = np.asarray(b_logvar, dtype=np.float32)

    mu = np.dot(x_arr, w_m) + b_m
    logvar = np.dot(x_arr, w_lv) + b_lv
    return mu, logvar


def gaussian_reparameterize_cpu(
    mu: np.ndarray,
    logvar: np.ndarray,
    eps: np.ndarray,
) -> np.ndarray:
    """Continuous pathwise reparameterization kernel: z = mu + exp(0.5 * logvar) * eps."""
    mu_arr = np.asarray(mu, dtype=np.float32)
    lv_arr = np.asarray(logvar, dtype=np.float32)
    eps_arr = np.asarray(eps, dtype=np.float32)

    std = np.exp(0.5 * lv_arr)
    return mu_arr + std * eps_arr


def gaussian_kl_divergence_cpu(
    mu: np.ndarray,
    logvar: np.ndarray,
) -> np.ndarray:
    """Analytical KL divergence between diagonal Gaussian and unit Gaussian N(0, I)."""
    mu_arr = np.asarray(mu, dtype=np.float32)
    lv_arr = np.asarray(logvar, dtype=np.float32)

    return -0.5 * np.sum(1.0 + lv_arr - mu_arr**2 - np.exp(lv_arr), axis=-1)


def linear_decode_cpu(
    z: np.ndarray,
    w_dec: np.ndarray,
    b_dec: np.ndarray,
) -> np.ndarray:
    """Vectorized linear decoder: z @ w_dec + b_dec."""
    z_arr = np.asarray(z, dtype=np.float32)
    w_d = np.asarray(w_dec, dtype=np.float32)
    b_d = np.asarray(b_dec, dtype=np.float32)

    return np.dot(z_arr, w_d) + b_d


def gumbel_softmax_sample_cpu(
    x: np.ndarray,
    w_logits: np.ndarray,
    b_logits: np.ndarray,
    gumbel_noise: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Vectorized Gumbel-Softmax relaxed categorical sampling."""
    x_arr = np.asarray(x, dtype=np.float32)
    w_l = np.asarray(w_logits, dtype=np.float32)
    b_l = np.asarray(b_logits, dtype=np.float32)
    g = np.asarray(gumbel_noise, dtype=np.float32)

    logits = np.dot(x_arr, w_l) + b_l
    unnorm = (logits + g) / float(temperature)
    max_val = np.max(unnorm, axis=-1, keepdims=True)
    exp_vals = np.exp(unnorm - max_val)
    return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)


def bernoulli_reinforce_score_cpu(
    probs: np.ndarray,
    b: np.ndarray,
    reward: np.ndarray,
    baseline: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Vectorized CPU score function gradient estimator for Bernoulli latents."""
    p = np.asarray(probs, dtype=np.float32)
    b_arr = np.asarray(b, dtype=np.float32)
    r = np.asarray(reward, dtype=np.float32)
    base = np.asarray(baseline, dtype=np.float32)

    adv = r - base
    if adv.ndim == 1:
        adv = np.expand_dims(adv, axis=-1)
    return adv * (b_arr - p)
