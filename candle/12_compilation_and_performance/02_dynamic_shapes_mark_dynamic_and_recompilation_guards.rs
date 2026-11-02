//! Dynamic Shapes Mark Dynamic and Recompilation Guards
//! in candle/12_compilation_and_performance.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Masked mean pooling across variable sequence lengths.
pub fn dynamic_masked_pool(x: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let x_vec = x.to_vec3::<f32>()?;
    let m_vec = mask.to_vec2::<f32>()?;
    let b = x_vec.len();
    let l = x_vec[0].len();
    let d = x_vec[0][0].len();

    let mut out = vec![vec![0.0f32; d]; b];
    for i in 0..b {
        let mut length = 0.0f32;
        for t in 0..l {
            let m = m_vec[i][t];
            length += m;
            for j in 0..d {
                out[i][j] += x_vec[i][t][j] * m;
            }
        }
        let denom = length.max(1.0f32);
        for j in 0..d {
            out[i][j] /= denom;
        }
    }
    Tensor::new(out, x.device())
}

/// Assign sequence lengths to the smallest enclosing bucket boundary.
pub fn bucket_allocation(seq_lens: &Tensor, bucket_boundaries: &Tensor) -> Result<Tensor> {
    let sl = seq_lens.to_vec1::<f32>()?;
    let bb = bucket_boundaries.to_vec1::<f32>()?;
    let n = sl.len();

    let mut allocated = vec![0.0f32; n];
    for i in 0..n {
        let len = sl[i];
        let mut best_b = f32::INFINITY;
        for &b_val in &bb {
            if b_val >= len && b_val < best_b {
                best_b = b_val;
            }
        }
        allocated[i] = best_b;
    }
    Tensor::new(allocated, seq_lens.device())
}

/// Root Mean Square Normalization (Zhang & Sennrich 2019) across dynamic features.
pub fn dynamic_rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let x_vec = x.to_vec2::<f32>()?;
    let w_vec = weight.to_vec1::<f32>()?;
    let b = x_vec.len();
    let d = x_vec[0].len();

    let mut out = vec![vec![0.0f32; d]; b];
    for i in 0..b {
        let mut sum_sq = 0.0f32;
        for j in 0..d {
            sum_sq += x_vec[i][j] * x_vec[i][j];
        }
        let var = sum_sq / (d as f32);
        let inv_rms = 1.0f32 / (var + eps).sqrt();
        for j in 0..d {
            out[i][j] = (x_vec[i][j] * inv_rms) * w_vec[j];
        }
    }
    Tensor::new(out, x.device())
}
