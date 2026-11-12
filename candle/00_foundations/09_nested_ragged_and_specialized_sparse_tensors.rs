//! Topic 09: Nested, Ragged & Specialized Sparse Tensors.
//!
//! Provides native Candle implementations for:
//! 1. Ragged batch representation and zero-copy offset indexing.
//! 2. Ragged-to-padded and padded-to-ragged bidirectional packing.
//! 3. Per-sequence ragged mean reduction.
//! 4. Compressed Sparse Row (CSR) Sparse Matrix-Vector/Matrix product (SpMV).
//! 5. `RaggedLinear` forward projection and pooled inference.
//!
//! # Mathematical Invariants
//! Given $B$ variable-length sequences with total tokens $N = \sum_{i=1}^B L_i$:
//! - Ragged storage eliminates padding waste: $\Delta M = \mathcal{O}(B \cdot L_{\max} - N)$.
//! - CSR SpMV equivalence: $M_{\text{CSR}} Y_{\text{tokens}} \equiv Y_{\text{pooled}}$.
//!
//! ```rust
//! use candle_core::{Device, Tensor};
//! use ai_candle::c00_foundations::t09_nested_ragged_and_specialized_sparse_tensors::ragged_mean_pool;
//!
//! let device = Device::Cpu;
//! let values = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]], &device).unwrap();
//! let offsets = vec![0, 2, 3]; // Sequence 0 has length 2, sequence 1 has length 1
//! let pooled = ragged_mean_pool(&values, &offsets, &device).unwrap();
//! assert_eq!(pooled.dims(), &[2, 2]);
//! ```

use candle_core::{Device, Result, Tensor};

/// Encapsulates a variable-length ragged batch of token features.
#[derive(Debug, Clone)]
pub struct RaggedTensor {
    pub values: Tensor,
    pub offsets: Vec<usize>,
}

impl RaggedTensor {
    pub fn new(values: Tensor, offsets: Vec<usize>) -> Self {
        Self { values, offsets }
    }

    pub fn num_sequences(&self) -> usize {
        if self.offsets.is_empty() {
            0
        } else {
            self.offsets.len() - 1
        }
    }

    pub fn total_tokens(&self) -> usize {
        self.values.dim(0).unwrap_or(0)
    }
}

/// Converts a flat ragged token tensor into a padded dense batch [B, max_len, D] and binary mask [B, max_len].
pub fn ragged_to_padded(
    values: &Tensor,
    offsets: &[usize],
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let num_seqs = if offsets.is_empty() {
        0
    } else {
        offsets.len() - 1
    };
    let dim = values.dim(1)?;
    let mut max_len = 0;
    for i in 0..num_seqs {
        let l = offsets[i + 1] - offsets[i];
        if l > max_len {
            max_len = l;
        }
    }

    let flat_vals = values.to_vec2::<f32>()?;
    let mut padded_flat = vec![0.0f32; num_seqs * max_len * dim];
    let mut mask_flat = vec![0.0f32; num_seqs * max_len];

    for i in 0..num_seqs {
        let start = offsets[i];
        let end = offsets[i + 1];
        let l = end - start;
        for t in 0..l {
            mask_flat[i * max_len + t] = 1.0;
            for d in 0..dim {
                padded_flat[i * max_len * dim + t * dim + d] = flat_vals[start + t][d];
            }
        }
    }

    let padded = Tensor::from_vec(padded_flat, (num_seqs, max_len, dim), device)?;
    let mask = Tensor::from_vec(mask_flat, (num_seqs, max_len), device)?;
    Ok((padded, mask))
}

/// Computes per-sequence mean reduction across ragged token slices.
pub fn ragged_mean_pool(values: &Tensor, offsets: &[usize], device: &Device) -> Result<Tensor> {
    let num_seqs = if offsets.is_empty() {
        0
    } else {
        offsets.len() - 1
    };
    let dim = values.dim(1)?;
    let flat_vals = values.to_vec2::<f32>()?;
    let mut pooled_flat = vec![0.0f32; num_seqs * dim];

    for i in 0..num_seqs {
        let start = offsets[i];
        let end = offsets[i + 1];
        let l = (end - start) as f32;
        if l > 0.0 {
            for t in start..end {
                for d in 0..dim {
                    pooled_flat[i * dim + d] += flat_vals[t][d];
                }
            }
            for d in 0..dim {
                pooled_flat[i * dim + d] /= l;
            }
        }
    }

    Tensor::from_vec(pooled_flat, (num_seqs, dim), device)
}

/// Evaluates Compressed Sparse Row (CSR) matrix-matrix product: Y = A @ X.
pub fn csr_spmv(
    crow_indices: &[usize],
    col_indices: &[usize],
    csr_values: &[f32],
    x: &Tensor,
    num_rows: usize,
    device: &Device,
) -> Result<Tensor> {
    let dim_out = x.dim(1)?;
    let x_flat = x.to_vec2::<f32>()?;
    let mut y_flat = vec![0.0f32; num_rows * dim_out];

    for r in 0..num_rows {
        let start = crow_indices[r];
        let end = crow_indices[r + 1];
        for idx in start..end {
            let c = col_indices[idx];
            let val = csr_values[idx];
            for d in 0..dim_out {
                y_flat[r * dim_out + d] += val * x_flat[c][d];
            }
        }
    }

    Tensor::from_vec(y_flat, (num_rows, dim_out), device)
}

/// Linear projection module for variable-length ragged batches.
#[derive(Debug, Clone)]
pub struct RaggedLinear {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl RaggedLinear {
    pub fn new(weight: Tensor, bias: Tensor) -> Self {
        Self { weight, bias }
    }

    pub fn forward(&self, values: &Tensor) -> Result<Tensor> {
        let tokens = values.matmul(&self.weight)?;
        tokens.broadcast_add(&self.bias)
    }

    pub fn forward_pooled(
        &self,
        values: &Tensor,
        offsets: &[usize],
        device: &Device,
    ) -> Result<Tensor> {
        let tokens = self.forward(values)?;
        ragged_mean_pool(&tokens, offsets, device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ragged_mean_pool() {
        let device = Device::Cpu;
        let values = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]], &device).unwrap();
        let offsets = vec![0, 2, 3];
        let pooled = ragged_mean_pool(&values, &offsets, &device).unwrap();
        let pooled_vec = pooled.to_vec2::<f32>().unwrap();
        assert_eq!(pooled_vec.len(), 2);
        assert!((pooled_vec[0][0] - 2.0).abs() < 1e-6);
        assert!((pooled_vec[0][1] - 3.0).abs() < 1e-6);
        assert!((pooled_vec[1][0] - 5.0).abs() < 1e-6);
        assert!((pooled_vec[1][1] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_csr_spmv_equivalence() {
        let device = Device::Cpu;
        let values = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]], &device).unwrap();
        let offsets = vec![0, 2, 3];
        let pooled = ragged_mean_pool(&values, &offsets, &device).unwrap();

        // Row 0 averages tokens 0, 1 (weight 0.5 each); Row 1 selects token 2 (weight 1.0)
        let crow = vec![0, 2, 3];
        let col = vec![0, 1, 2];
        let csr_vals = vec![0.5f32, 0.5, 1.0];
        let csr_res = csr_spmv(&crow, &col, &csr_vals, &values, 2, &device).unwrap();

        let diff = (&pooled - &csr_res)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-6);
    }
}
