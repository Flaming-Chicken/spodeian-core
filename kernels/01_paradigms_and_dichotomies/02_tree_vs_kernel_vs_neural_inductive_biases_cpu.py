"""Pure-CPU vectorized hardware kernels for Category 01 Topic 02:
02_tree_vs_kernel_vs_neural_inductive_biases
"""

from __future__ import annotations

import numpy as np


def soft_decision_tree_cpu(
    x: np.ndarray,
    routing_weights: np.ndarray,
    thresholds: np.ndarray,
    temperature: float,
    leaf_weights: np.ndarray,
) -> np.ndarray:
    """CPU SIMD-aligned kernel evaluating depth-2 differentiable soft decision tree.

    Args:
        x: Input matrix [batch, in_dim].
        routing_weights: Routing projection matrix [3, in_dim].
        thresholds: Node split thresholds [3].
        temperature: Inverse temperature scaling beta.
        leaf_weights: Leaf output weights [4, out_dim].

    Returns:
        Predicted values [batch, out_dim].
    """
    x_arr = np.asarray(x, dtype=np.float32)
    w_route = np.asarray(routing_weights, dtype=np.float32)
    thresh = np.asarray(thresholds, dtype=np.float32)
    w_leaf = np.asarray(leaf_weights, dtype=np.float32)

    logits = float(temperature) * (np.dot(x_arr, w_route.T) - thresh)
    p_right = 1.0 / (1.0 + np.exp(-logits))
    p_left = 1.0 - p_right

    mu0 = p_left[:, 0:1] * p_left[:, 1:2]
    mu1 = p_left[:, 0:1] * p_right[:, 1:2]
    mu2 = p_right[:, 0:1] * p_left[:, 2:3]
    mu3 = p_right[:, 0:1] * p_right[:, 2:3]
    leaf_probs = np.concatenate([mu0, mu1, mu2, mu3], axis=-1)

    return np.dot(leaf_probs, w_leaf)


def rbf_kernel_ridge_cpu(
    x: np.ndarray,
    support_vectors: np.ndarray,
    dual_coeffs: np.ndarray,
    gamma: float,
    bias: np.ndarray,
) -> np.ndarray:
    """CPU vectorized kernel evaluating RBF kernel expansion.

    Args:
        x: Input matrix [batch, in_dim].
        support_vectors: Training support points [support, in_dim].
        dual_coeffs: Dual expansion weights [support, out_dim].
        gamma: Kernel inverse bandwidth parameter.
        bias: Output additive bias [out_dim].

    Returns:
        Predicted values [batch, out_dim].
    """
    x_arr = np.asarray(x, dtype=np.float32)
    supp = np.asarray(support_vectors, dtype=np.float32)
    coeffs = np.asarray(dual_coeffs, dtype=np.float32)
    b = np.asarray(bias, dtype=np.float32)

    x_sq = np.sum(x_arr**2, axis=-1, keepdims=True)
    s_sq = np.sum(supp**2, axis=-1, keepdims=True).T
    dists_sq = np.maximum(x_sq + s_sq - 2.0 * np.dot(x_arr, supp.T), 0.0)
    sim = np.exp(-float(gamma) * dists_sq)

    return np.dot(sim, coeffs) + b


def mlp_classifier_cpu(
    x: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
) -> np.ndarray:
    """CPU vectorized kernel evaluating 2-layer ReLU MLP.

    Args:
        x: Input matrix [batch, in_dim].
        w1: Input weight matrix [in_dim, hidden_dim].
        b1: Hidden bias [hidden_dim].
        w2: Output weight matrix [hidden_dim, out_dim].
        b2: Output bias [out_dim].

    Returns:
        Predicted values [batch, out_dim].
    """
    x_arr = np.asarray(x, dtype=np.float32)
    w1_arr = np.asarray(w1, dtype=np.float32)
    b1_arr = np.asarray(b1, dtype=np.float32)
    w2_arr = np.asarray(w2, dtype=np.float32)
    b2_arr = np.asarray(b2, dtype=np.float32)

    h = np.maximum(np.dot(x_arr, w1_arr) + b1_arr, 0.0)
    return np.dot(h, w2_arr) + b2_arr
