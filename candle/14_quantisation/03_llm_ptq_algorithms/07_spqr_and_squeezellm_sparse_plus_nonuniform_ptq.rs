//! SpQR and SqueezeLLM: Sparse Plus Nonuniform PTQ
//! in candle/14_quantisation/03_llm_ptq_algorithms.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Computes Fisher/Hessian diagonal sensitivity: S_ij = H_ii * |W_ij|.
pub fn compute_hessian_sensitivity(x_calib: &Tensor, weight: &Tensor) -> Result<(Tensor, Tensor)> {
    let (n_samples, d_in) = x_calib.dims2()?;
    let (_, d_out) = weight.dims2()?;
    let x_vec = x_calib.to_vec2::<f32>()?;
    let w_vec = weight.to_vec2::<f32>()?;

    let mut h_diag = vec![0.0f32; d_in];
    for col in 0..d_in {
        let mut sum_sq = 0.0f32;
        for row in 0..n_samples {
            let v = x_vec[row][col];
            sum_sq += v * v;
        }
        h_diag[col] = (2.0 * sum_sq) / (n_samples as f32);
    }

    let mut sensitivity_flat = vec![0.0f32; d_in * d_out];
    for i in 0..d_in {
        let h_ii = h_diag[i];
        for j in 0..d_out {
            sensitivity_flat[i * d_out + j] = h_ii * w_vec[i][j].abs();
        }
    }

    let h_diag_t = Tensor::from_vec(h_diag, (d_in,), x_calib.device())?;
    let sensitivity_t = Tensor::from_vec(sensitivity_flat, (d_in, d_out), weight.device())?;
    Ok((h_diag_t, sensitivity_t))
}

/// Isolates the top-k most sensitive weights into a high-precision sparse component.
pub fn extract_sparse_outliers(
    weight: &Tensor,
    sensitivity: &Tensor,
    top_k_count: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (d_in, d_out) = weight.dims2()?;
    let w_vec = weight.to_vec2::<f32>()?;
    let s_flat = sensitivity
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect::<Vec<f32>>();

    let mut sorted_sens = s_flat.clone();
    sorted_sens.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = sorted_sens[sorted_sens.len() - top_k_count];

    let mut w_sparse_flat = vec![0.0f32; d_in * d_out];
    let mut w_dense_flat = vec![0.0f32; d_in * d_out];
    let mut mask_flat = vec![0.0f32; d_in * d_out];

    for i in 0..d_in {
        for j in 0..d_out {
            let idx = i * d_out + j;
            let val = w_vec[i][j];
            let sens = s_flat[idx];
            if sens >= threshold {
                w_sparse_flat[idx] = val;
                mask_flat[idx] = 1.0;
            } else {
                w_dense_flat[idx] = val;
            }
        }
    }

    let w_sparse = Tensor::from_vec(w_sparse_flat, (d_in, d_out), weight.device())?;
    let w_dense = Tensor::from_vec(w_dense_flat, (d_in, d_out), weight.device())?;
    let sparse_mask = Tensor::from_vec(mask_flat, (d_in, d_out), weight.device())?;
    Ok((w_sparse, w_dense, sparse_mask))
}

/// Symmetric 3-bit quantization of the dense non-outlier weights.
pub fn quantize_dense_int3(
    w_dense: &Tensor,
    sparse_mask: &Tensor,
    qmin: f32,
    qmax: f32,
) -> Result<(Tensor, Tensor)> {
    let (d_in, d_out) = w_dense.dims2()?;
    let wd_vec = w_dense.to_vec2::<f32>()?;
    let mask_vec = sparse_mask.to_vec2::<f32>()?;

    let mut scales = vec![0.0f32; d_out];
    let mut deq_flat = vec![0.0f32; d_in * d_out];

    for j in 0..d_out {
        let mut max_abs = 0.0f32;
        for i in 0..d_in {
            let v = wd_vec[i][j].abs();
            if v > max_abs {
                max_abs = v;
            }
        }
        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / qmax };
        scales[j] = scale;

        for i in 0..d_in {
            let orig = wd_vec[i][j];
            if mask_vec[i][j] > 0.0 {
                deq_flat[i * d_out + j] = 0.0;
            } else {
                let q = (orig / scale).round().clamp(qmin, qmax);
                deq_flat[i * d_out + j] = q * scale;
            }
        }
    }

    let deq_t = Tensor::from_vec(deq_flat, (d_in, d_out), w_dense.device())?;
    let scale_t = Tensor::from_vec(scales, (1, d_out), w_dense.device())?;
    Ok((deq_t, scale_t))
}

/// SpQR forward pass: Y = X @ W_dense + X @ W_sparse + bias.
pub fn spqr_forward(
    x: &Tensor,
    w_dense_deq: &Tensor,
    w_sparse: &Tensor,
    bias: Option<&Tensor>,
) -> Result<(Tensor, Tensor, Tensor)> {
    let y_dense = x.matmul(w_dense_deq)?;
    let y_sparse = x.matmul(w_sparse)?;
    let mut y_total = (&y_dense + &y_sparse)?;
    if let Some(bi) = bias {
        y_total = y_total.broadcast_add(bi)?;
    }
    Ok((y_total, y_dense, y_sparse))
}
