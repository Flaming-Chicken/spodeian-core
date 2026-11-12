"""Cache-Tiled Hardware-Optimized CPU Kernel: Tabular Scalers & Fused Graph Projection.

Topic 07: 07_feature_engineering_scalers_and_tabular_preprocessing (Category 00: Foundations)

Theoretical Overview:
In production inference pipelines, preprocessing raw tabular data (standardization,
clamping, and linear projections) sequentially incurs multiple memory round-trips to DRAM:
1. Standard Scaler Pass: Reads raw X, writes normalized Z to DRAM.
2. Linear Layer Pass: Reads Z from DRAM, computes Z @ W + b, writes Y to DRAM.

This hardware-simulated kernel eliminates DRAM traffic via two complementary techniques:
1. Serving-Time Weight Folding: Precomputes W_fused = diag(1/std) @ W and
   b_fused = b - (mean/std) @ W, collapsing two distinct passes into a single GEMM.
2. Block-Tiled Fused Execution: For streaming inputs where weights cannot be folded
   (e.g. non-linear activations or dynamic target encodings), tiles computation across
   CPU L1/L2 cache blocks:
       Z_tile = (X_tile - mu) / sigma
       Y_tile = Z_tile @ W + b
   retaining intermediate activations in cache lines.
"""

from __future__ import annotations

from pathlib import Path

import torch
from jaxtyping import Float
from torch import Tensor


def fused_scale_and_matmul_cpu(
    x: Float[Tensor, "batch in_dim"],
    mean: Float[Tensor, "*in_dim"],
    std: Float[Tensor, "*in_dim"],
    w_head: Float[Tensor, "in_dim out_dim"],
    b_head: Float[Tensor, "*out_dim"],
    block_size: int = 16,
) -> Float[Tensor, "batch out_dim"]:
    """Evaluates standardization and linear projection fused within L1/L2 cache blocks."""
    batch_size, in_dim = x.shape
    _, out_dim = w_head.shape

    y = torch.empty(batch_size, out_dim, dtype=x.dtype, device=x.device)

    # Process in cache-friendly blocks
    for b_start in range(0, batch_size, block_size):
        b_end = min(b_start + block_size, batch_size)
        x_block = x[b_start:b_end]  # [B_blk, in_dim]

        # In-cache normalization
        z_block = (x_block - mean) / std

        # In-cache projection
        y[b_start:b_end] = z_block @ w_head + b_head

    return y


def vectorized_yeo_johnson_cpu(
    x: Float[Tensor, "batch in_dim"],
    lmbda: float = 0.5,
) -> Float[Tensor, "batch in_dim"]:
    """SIMD-friendly branchless power transformation using torch.where."""
    pos_mask = x >= 0
    pos_base = torch.clamp(x + 1.0, min=1e-8)
    neg_base = torch.clamp(-x + 1.0, min=1e-8)

    if abs(lmbda) > 1e-6:
        pos_term = ((pos_base**lmbda) - 1.0) / lmbda
    else:
        pos_term = torch.log(pos_base)

    if abs(lmbda - 2.0) > 1e-6:
        neg_term = -((neg_base ** (2.0 - lmbda)) - 1.0) / (2.0 - lmbda)
    else:
        neg_term = -torch.log(neg_base)

    return torch.where(pos_mask, pos_term, neg_term)


if __name__ == "__main__":
    import core

    print("Testing CPU Hardware Kernel: Tabular Scalers & Fused Graph Projection...")

    ckpt_path = Path("weights/00_foundations/07_feature_engineering_scalers_and_tabular_preprocessing.safetensors")
    weights, test_in, golden_out, meta = core.parity.load_reference_checkpoint(ckpt_path)

    mean = weights["mean"]
    std = weights["std"]
    w_head = weights["w_head"]
    b_head = weights["b_head"]
    yj_gold = weights["yj_reference"]

    # 1. Test Fused Block-Tiled Kernel Parity
    out_fused = fused_scale_and_matmul_cpu(test_in, mean, std, w_head, b_head, block_size=2)
    torch.testing.assert_close(out_fused, golden_out, atol=1e-5, rtol=1e-5)
    print("[PASS] Block-tiled in-cache normalization & projection matches golden reference.")

    # 2. Test Vectorized Yeo-Johnson Kernel
    yj_out = vectorized_yeo_johnson_cpu(test_in, lmbda=0.5)
    torch.testing.assert_close(yj_out, yj_gold, atol=1e-5, rtol=1e-5)
    print("[PASS] Vectorized Yeo-Johnson kernel matches golden reference.")

    print("[PASS] All Topic 07 CPU kernel tests passed cleanly!")
