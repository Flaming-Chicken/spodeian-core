"""Topic 02: Fused Strided Sliding-Window GEMM Kernel (CPU Simulator).

Demonstrates the core mechanics of memory strides and implicit GEMM (im2col fusion):
In naive implementations, sliding window extraction (e.g. unfold / im2col) explicitly
allocates an expanded intermediate tensor in RAM:
    Input: [B, L]
    Unfolded Window Buffer: [B, Num_Windows, Window_Size] -> (Memory Bloat: up to 10x!)
    Projection: Unfolded @ W

This fused kernel performs implicit strided GEMM directly from the source buffer,
computing output elements on-the-fly via linear stride arithmetic:
    out[b, w, o] = sum_k (x[b * s_b + (w * step + k) * s_l] * W[k, o])
achieving zero intermediate buffer allocations and maximum cache locality.
"""

from __future__ import annotations

import time

import torch


def naive_unfold_and_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    window_size: int,
    step: int = 1,
) -> torch.Tensor:
    """Naive two-pass implementation: materializes full unfolded tensor then performs GEMM."""
    batch_size, length = x.shape
    num_windows = (length - window_size) // step + 1
    # Materialize intermediate unfolded tensor
    unfolded = torch.empty((batch_size, num_windows, window_size), dtype=x.dtype, device=x.device)
    for w in range(num_windows):
        unfolded[:, w, :] = x[:, w * step : w * step + window_size]

    # GEMM: [B, W, K] @ [K, Out] -> [B, W, Out]
    return torch.matmul(unfolded, weight)


def fused_strided_gemm_cpu(
    x: torch.Tensor,
    weight: torch.Tensor,
    window_size: int,
    step: int = 1,
) -> torch.Tensor:
    """Fused implementation using as_strided view to perform GEMM without intermediate allocation."""
    batch_size, length = x.shape
    num_windows = (length - window_size) // step + 1
    batch_stride, elem_stride = x.stride()

    # Zero-copy strided view
    strided_view = torch.as_strided(
        x,
        size=(batch_size, num_windows, window_size),
        stride=(batch_stride, elem_stride * step, elem_stride),
    )
    # Direct GEMM on the strided view
    return torch.matmul(strided_view, weight)


def benchmark_strided_vs_unfold(
    batch_size: int = 64,
    seq_len: int = 1024,
    window_size: int = 32,
    out_dim: int = 64,
    iters: int = 50,
) -> dict[str, float]:
    """Profiles execution time and speedup between naive unfold and strided GEMM."""
    x = torch.randn(batch_size, seq_len, dtype=torch.float32)
    w = torch.randn(window_size, out_dim, dtype=torch.float32)

    # Warmup
    _ = naive_unfold_and_gemm(x, w, window_size=window_size)
    _ = fused_strided_gemm_cpu(x, w, window_size=window_size)

    # Time naive unfold
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = naive_unfold_and_gemm(x, w, window_size=window_size)
    dur_naive = (time.perf_counter() - t0) / iters * 1000.0

    # Time strided view
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = fused_strided_gemm_cpu(x, w, window_size=window_size)
    dur_strided = (time.perf_counter() - t0) / iters * 1000.0

    return {
        "naive_unfold_ms": dur_naive,
        "strided_view_ms": dur_strided,
        "speedup": dur_naive / max(dur_strided, 1e-9),
    }


if __name__ == "__main__":
    print("Testing CPU Strided Sliding-Window GEMM Kernel...")
    b, seq_len, k, out_dim = 4, 16, 4, 2
    x_test = torch.randn(b, seq_len)
    w_test = torch.randn(k, out_dim)

    out_naive = naive_unfold_and_gemm(x_test, w_test, window_size=k, step=2)
    out_fused = fused_strided_gemm_cpu(x_test, w_test, window_size=k, step=2)

    diff = float(torch.norm(out_naive - out_fused, p=float("inf")).item())
    print(f"Max Discrepancy between Naive Unfold & Strided View: {diff:.2e}")
    assert diff < 1e-5, f"Discrepancy {diff} exceeds 1e-5"

    stats = benchmark_strided_vs_unfold()
    print(
        f"Naive Unfold: {stats['naive_unfold_ms']:.3f} ms | "
        f"Strided View: {stats['strided_view_ms']:.3f} ms | "
        f"Speedup: {stats['speedup']:.2f}x"
    )
    print("[PASS] CPU Strided Kernel validation complete.")
