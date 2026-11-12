//! Model Merging Slerp Task Arithmetic TIES DARE and Evolutionary
//! in candle/15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Spherical Linear Interpolation (SLERP) between weight tensors.
pub fn slerp_merge(v0: &Tensor, v1: &Tensor, t: f32) -> Result<Tensor> {
    let shape = v0.shape();
    let v0_vec = v0.flatten_all()?.to_vec1::<f32>()?;
    let v1_vec = v1.flatten_all()?.to_vec1::<f32>()?;
    let n = v0_vec.len();

    let norm0: f32 = v0_vec.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm1: f32 = v1_vec.iter().map(|&x| x * x).sum::<f32>().sqrt();

    if norm0 < 1e-6 || norm1 < 1e-6 {
        let mut out = vec![0.0f32; n];
        for i in 0..n {
            out[i] = (1.0 - t) * v0_vec[i] + t * v1_vec[i];
        }
        return Tensor::from_vec(out, shape, v0.device());
    }

    let mut dot = 0.0f32;
    for i in 0..n {
        dot += (v0_vec[i] / norm0) * (v1_vec[i] / norm1);
    }
    let dot = dot.clamp(-1.0, 1.0);
    let theta = dot.acos();

    if theta.sin().abs() < 1e-6 || theta.abs() < 1e-6 {
        let mut out = vec![0.0f32; n];
        for i in 0..n {
            out[i] = (1.0 - t) * v0_vec[i] + t * v1_vec[i];
        }
        return Tensor::from_vec(out, shape, v0.device());
    }

    let sin_theta = theta.sin();
    let w0 = ((1.0 - t) * theta).sin() / sin_theta;
    let w1 = (t * theta).sin() / sin_theta;

    let mut out = vec![0.0f32; n];
    for i in 0..n {
        out[i] = w0 * v0_vec[i] + w1 * v1_vec[i];
    }

    Tensor::from_vec(out, shape, v0.device())
}

/// Task arithmetic parameter addition: theta = theta_base + sum_k w_k * delta_k.
pub fn task_arithmetic_merge(base: &Tensor, deltas: &Tensor, weights: &Tensor) -> Result<Tensor> {
    let (num_tasks, _, _) = deltas.dims3()?;
    let w_vec = weights.flatten_all()?.to_vec1::<f32>()?;
    let mut merged = base.clone();

    for k in 0..num_tasks {
        let d_k = deltas.get(k)?;
        let scaled_d = (d_k * (w_vec[k] as f64))?;
        merged = (merged + scaled_d)?;
    }

    Ok(merged)
}

/// TIES-Merging (Trimming, Electing Sign, and Disjoint Merge).
pub fn ties_merge(base: &Tensor, deltas: &Tensor, trim_ratio: f32, scale: f32) -> Result<Tensor> {
    let (num_tasks, dim1, dim2) = deltas.dims3()?;
    let total_elements = dim1 * dim2;
    let keep_k = ((total_elements as f32 * (1.0 - trim_ratio)).round() as usize).max(1);

    let d_vec = deltas.flatten_all()?.to_vec1::<f32>()?;
    let mut trimmed_deltas = vec![0.0f32; num_tasks * total_elements];

    // 1. Trimming step
    for m in 0..num_tasks {
        let m_offset = m * total_elements;
        let mut indexed: Vec<(usize, f32)> = (0..total_elements)
            .map(|i| (i, d_vec[m_offset + i].abs()))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        for &(idx, _) in indexed.iter().take(keep_k) {
            trimmed_deltas[m_offset + idx] = d_vec[m_offset + idx];
        }
    }

    // 2. Elect sign
    let mut elected_sign = vec![0.0f32; total_elements];
    for i in 0..total_elements {
        let mut sum_d = 0.0f32;
        for m in 0..num_tasks {
            sum_d += trimmed_deltas[m * total_elements + i];
        }
        elected_sign[i] = if sum_d > 0.0 {
            1.0
        } else if sum_d < 0.0 {
            -1.0
        } else {
            0.0
        };
    }

    // 3. Disjoint merge
    let mut merged_delta = vec![0.0f32; total_elements];
    let mut counts = vec![0.0f32; total_elements];

    for m in 0..num_tasks {
        let m_offset = m * total_elements;
        for i in 0..total_elements {
            let val = trimmed_deltas[m_offset + i];
            let sign_val = if val > 0.0 {
                1.0
            } else if val < 0.0 {
                -1.0
            } else {
                0.0
            };
            if sign_val == elected_sign[i] && val != 0.0 {
                merged_delta[i] += val;
                counts[i] += 1.0;
            }
        }
    }

    let mut avg_delta = vec![0.0f32; total_elements];
    for i in 0..total_elements {
        if counts[i] > 0.0 {
            avg_delta[i] = merged_delta[i] / counts[i];
        }
    }

    let avg_tensor = Tensor::from_vec(avg_delta, (dim1, dim2), base.device())?;
    base.broadcast_add(&(avg_tensor * (scale as f64))?)
}

/// DARE (Drop And REscale) delta transformation.
pub fn dare_rescale(delta: &Tensor, mask: &Tensor, drop_prob: f32) -> Result<Tensor> {
    let rescale_factor = 1.0 / (1.0 - drop_prob);
    let scaled_delta = (delta * (rescale_factor as f64))?;
    scaled_delta.broadcast_mul(mask)
}
