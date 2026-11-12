"""Pure-CPU vectorized hardware kernels for Category 01 Topic 01:
01_generative_vs_discriminative_classifiers
"""

from __future__ import annotations

import numpy as np


def fit_gda_parameters_cpu(
    x: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    reg_lambda: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """CPU kernel fitting Gaussian Discriminant Analysis (GDA) with shared covariance."""
    x_mat = np.asarray(x, dtype=np.float32)
    y_vec = np.asarray(y, dtype=np.int64)
    num_samples, dim = x_mat.shape

    means = np.zeros((num_classes, dim), dtype=np.float32)
    priors = np.zeros(num_classes, dtype=np.float32)

    for c in range(num_classes):
        mask = y_vec == c
        count = float(np.sum(mask))
        priors[c] = count / float(num_samples)
        means[c] = np.mean(x_mat[mask], axis=0)

    centered_x = x_mat.copy()
    for c in range(num_classes):
        mask = y_vec == c
        centered_x[mask] = x_mat[mask] - means[c]

    shared_cov = np.dot(centered_x.T, centered_x) / float(num_samples)
    reg_cov = shared_cov + reg_lambda * np.eye(dim, dtype=np.float32)
    cov_inv = np.linalg.inv(reg_cov)

    w_derived = np.zeros((dim, num_classes), dtype=np.float32)
    b_derived = np.zeros(num_classes, dtype=np.float32)

    for c in range(num_classes):
        mu_c = means[c]
        sigma_inv_mu = np.dot(cov_inv, mu_c)
        w_derived[:, c] = sigma_inv_mu
        b_derived[c] = -0.5 * float(np.dot(mu_c, sigma_inv_mu)) + np.log(priors[c] + 1e-12)

    return means, reg_cov, priors, w_derived, b_derived


def predict_logits_gda_cpu(
    x: np.ndarray,
    w_derived: np.ndarray,
    b_derived: np.ndarray,
) -> np.ndarray:
    """CPU kernel evaluating generative GDA affine logits."""
    return np.dot(x, w_derived) + b_derived


def predict_proba_softmax_cpu(logits: np.ndarray) -> np.ndarray:
    """CPU kernel evaluating numerically stable softmax."""
    m = np.max(logits, axis=-1, keepdims=True)
    exp_shifted = np.exp(logits - m)
    return exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)


def discriminative_forward_cpu(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """CPU kernel evaluating discriminative linear forward pass."""
    return np.dot(x, weight) + bias
