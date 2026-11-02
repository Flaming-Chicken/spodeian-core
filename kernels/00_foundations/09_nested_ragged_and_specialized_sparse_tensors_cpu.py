"""Pure-CPU tiled/vectorized hardware kernels for Topic 09:
09_nested_ragged_and_specialized_sparse_tensors
"""

from __future__ import annotations

import numpy as np


def ragged_mean_pool_cpu(
    values: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    """Computes per-sequence mean reduction with cache-line sequential memory reads."""
    num_seqs = len(offsets) - 1
    dim = values.shape[1]
    pooled = np.zeros((num_seqs, dim), dtype=np.float32)

    for i in range(num_seqs):
        start = int(offsets[i])
        end = int(offsets[i + 1])
        length = float(end - start)
        if length > 0.0:
            # Vectorized contiguous slice sum
            pooled[i] = np.sum(values[start:end], axis=0) / length

    return pooled


def csr_spmv_cpu(
    crow_indices: np.ndarray,
    col_indices: np.ndarray,
    csr_values: np.ndarray,
    x: np.ndarray,
    num_rows: int,
) -> np.ndarray:
    """Compressed Sparse Row (CSR) SpMV kernel with zero-copy row traversals."""
    dim_out = x.shape[1]
    y = np.zeros((num_rows, dim_out), dtype=np.float32)

    for r in range(num_rows):
        start = int(crow_indices[r])
        end = int(crow_indices[r + 1])
        row_acc = np.zeros(dim_out, dtype=np.float32)
        for idx in range(start, end):
            c = int(col_indices[idx])
            val = csr_values[idx]
            row_acc += val * x[c]
        y[r] = row_acc

    return y


def ragged_gemm_cpu(
    values: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """Direct BLAS-compatible GEMM on flattened contiguous ragged tokens."""
    return np.matmul(values, weight) + bias


def ragged_to_padded_cpu(
    values: np.ndarray,
    offsets: np.ndarray,
    padding_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Packs flattened ragged tokens into a padded dense batch [B, max_len, D] and boolean mask."""
    num_seqs = len(offsets) - 1
    seq_lens = [int(offsets[i + 1] - offsets[i]) for i in range(num_seqs)]
    max_len = max(seq_lens)
    dim = values.shape[1]

    padded = np.full((num_seqs, max_len, dim), fill_value=padding_value, dtype=np.float32)
    mask = np.zeros((num_seqs, max_len), dtype=bool)

    for i in range(num_seqs):
        start = int(offsets[i])
        end = int(offsets[i + 1])
        length = end - start
        padded[i, :length] = values[start:end]
        mask[i, :length] = True

    return padded, mask


def padded_to_ragged_cpu(
    padded: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extracts contiguous ragged tokens and offset vectors from a padded dense batch."""
    num_seqs = padded.shape[0]
    lengths = [int(np.sum(mask[i])) for i in range(num_seqs)]
    offsets = np.zeros(num_seqs + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)

    total_tokens = int(offsets[-1])
    dim = padded.shape[-1]
    values = np.zeros((total_tokens, dim), dtype=np.float32)

    for i in range(num_seqs):
        start = int(offsets[i])
        end = int(offsets[i + 1])
        values[start:end] = padded[i, : lengths[i]]

    return values, offsets
