"""Standardized quantitative verification metrics, numerical invariants, and parity evaluators."""

from __future__ import annotations

from typing import cast

import torch


def sqnr_db(orig: torch.Tensor, quant: torch.Tensor, eps: float = 1e-12) -> float:
    """Computes Signal-to-Quantisation-Noise Ratio (SQNR) in decibels (dB).

    SQNR = 10 * log10( ||orig||^2 / (||orig - quant||^2 + eps) )
    """
    orig_f = orig.to(dtype=torch.float64)
    quant_f = quant.to(dtype=torch.float64)
    sig_power = torch.sum(orig_f**2)
    noise_power = torch.sum((orig_f - quant_f) ** 2) + eps
    ratio = sig_power / noise_power
    return float(10.0 * torch.log10(ratio).item())


def effective_bits_per_weight(weight_tensor: torch.Tensor, eps: float = 1e-12) -> float:
    """Computes empirical Shannon entropy in bits per weight across discrete values."""
    flattened = weight_tensor.detach().flatten()
    unique_vals, counts = torch.unique(flattened, return_counts=True)
    probs = counts.float() / flattened.numel()
    entropy = -torch.sum(probs * torch.log2(probs + eps)).item()
    return float(entropy)


def expected_calibration_error(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 10,
) -> float:
    """Computes Expected Calibration Error (ECE) for multi-class or binary predictions.

    Args:
        probs: Predicted probability distributions [N, C] or confidence scores [N].
        labels: Ground truth class integer indices [N] or binary labels [N].
        n_bins: Number of equal-width confidence partitions in [0, 1].
    """
    if probs.ndim > 1 and probs.shape[-1] > 1:
        confidences, predictions = torch.max(probs, dim=-1)
    else:
        confidences = probs.squeeze()
        predictions = (confidences >= 0.5).long()

    labels = labels.squeeze().long()
    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1)
    ece = torch.tensor(0.0, dtype=torch.float64)
    total_samples = len(labels)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if i == 0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)

        bin_count = in_bin.sum().item()
        if bin_count > 0:
            bin_acc = (predictions[in_bin] == labels[in_bin]).float().mean()
            bin_conf = confidences[in_bin].float().mean()
            bin_error = torch.abs(bin_acc - bin_conf)
            ece += bin_error * (bin_count / total_samples)

    return float(ece.item())


def linear_cka(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-12) -> float:
    """Computes linear Centered Kernel Alignment (CKA) between representations [N, D1] and [N, D2]."""
    X = X.to(dtype=torch.float64)
    Y = Y.to(dtype=torch.float64)
    # Center activations across batch dimension
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # HSIC(X, Y) = ||Y^T X||_F^2 / (N - 1)^2
    dot_xy = torch.sum((X.t() @ Y) ** 2)
    dot_xx = torch.sum((X.t() @ X) ** 2)
    dot_yy = torch.sum((Y.t() @ Y) ** 2)

    denom = torch.sqrt(dot_xx * dot_yy) + eps
    return float((dot_xy / denom).item())


def effective_rank(matrix: torch.Tensor, eps: float = 1e-12) -> float:
    """Computes Roy & Vetterli effective rank = exp(Shannon entropy of normalized singular values)."""
    if matrix.ndim != 2:
        raise ValueError("Matrix must be 2D")
    s = torch.linalg.svdvals(matrix.to(dtype=torch.float64))
    s_sum = torch.sum(s)
    if s_sum <= eps:
        return 0.0
    p = s / s_sum
    # Shannon entropy with natural log
    p_nz = p[p > eps]
    entropy = -torch.sum(p_nz * torch.log(p_nz))
    return float(torch.exp(entropy).item())


def simplex_etf_error(
    features: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int,
    eps: float = 1e-12,
) -> float:
    """Measures deviation from an ideal Simplex Equiangular Tight Frame (Neural Collapse NC2).

    Ideal mutual angles: cos(theta_{i, j}) = -1 / (K - 1) for i != j.
    """
    d = features.shape[-1]
    means = torch.zeros(n_classes, d, dtype=torch.float64)
    for c in range(n_classes):
        mask = labels.squeeze() == c
        if mask.any():
            means[c] = features[mask].to(dtype=torch.float64).mean(dim=0)

    # Center around global mean
    global_mean = means.mean(dim=0, keepdim=True)
    centered = means - global_mean
    norms = torch.norm(centered, p=2, dim=1, keepdim=True) + eps
    normalized = centered / norms

    # Pairwise cosines
    cosines = normalized @ normalized.t()
    ideal_cosine = -1.0 / (n_classes - 1) if n_classes > 1 else 0.0

    # Off-diagonal error
    off_diag_mask = ~torch.eye(n_classes, dtype=torch.bool)
    if off_diag_mask.any():
        err = torch.mean(torch.abs(cosines[off_diag_mask] - ideal_cosine))
        return float(err.item())
    return 0.0


def relative_l2_error(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-12) -> float:
    """Computes relative L2 Frobenius error: ||y_true - y_pred||_2 / (||y_true||_2 + eps)."""
    diff = y_true.to(dtype=torch.float64) - y_pred.to(dtype=torch.float64)
    norm_diff = torch.norm(diff, p=2)
    norm_true = torch.norm(y_true.to(dtype=torch.float64), p=2) + eps
    return float((norm_diff / norm_true).item())


def cosine_similarity_matrix(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Computes pairwise cosine similarity matrix between row vectors of a [M, D] and b [N, D]."""
    a_norm = a / (torch.norm(a, p=2, dim=-1, keepdim=True) + eps)
    b_norm = b / (torch.norm(b, p=2, dim=-1, keepdim=True) + eps)
    return cast(torch.Tensor, a_norm @ b_norm.t())


def wasserstein_1d(u_values: torch.Tensor, v_values: torch.Tensor) -> float:
    """Computes 1D Wasserstein-1 (Earth Mover's) distance between empirical distributions."""
    u_sorted, _ = torch.sort(u_values.detach().flatten().to(dtype=torch.float64))
    v_sorted, _ = torch.sort(v_values.detach().flatten().to(dtype=torch.float64))

    # Evaluate CDF difference along joint sorted values
    all_vals, _ = torch.sort(torch.cat([u_sorted, v_sorted]))
    u_cdf = torch.searchsorted(u_sorted, all_vals, right=True).double() / len(u_sorted)
    v_cdf = torch.searchsorted(v_sorted, all_vals, right=True).double() / len(v_sorted)

    deltas = all_vals[1:] - all_vals[:-1]
    cdf_diffs = torch.abs(u_cdf[:-1] - v_cdf[:-1])
    return float(torch.sum(deltas * cdf_diffs).item())
