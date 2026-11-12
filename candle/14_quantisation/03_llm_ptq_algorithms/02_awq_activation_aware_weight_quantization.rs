//! AWQ Activation Aware Weight Quantization
//! in candle/14_quantisation/03_llm_ptq_algorithms.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Standard Round-To-Nearest (RTN) per-column symmetric quantization.
pub fn rtn_quantize_per_column(weight: &Tensor, qmin: f32, qmax: f32) -> Result<(Tensor, Tensor)> {
    let (d_in, d_out) = weight.dims2()?;
    let w_vec = weight.to_vec2::<f32>()?;
    let mut out_vec = vec![0.0f32; d_in * d_out];
    let mut col_scales = vec![0.0f32; d_out];

    for j in 0..d_out {
        let mut max_abs = 0.0f32;
        for i in 0..d_in {
            let val = w_vec[i][j].abs();
            if val > max_abs {
                max_abs = val;
            }
        }
        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / qmax };
        col_scales[j] = scale;
        for i in 0..d_in {
            let orig = w_vec[i][j];
            let q = (orig / scale).round().clamp(qmin, qmax) * scale;
            out_vec[i * d_out + j] = q;
        }
    }

    let dequant = Tensor::from_vec(out_vec, (d_in, d_out), weight.device())?;
    let scales = Tensor::from_vec(col_scales, (1, d_out), weight.device())?;
    Ok((dequant, scales))
}

/// Computes AWQ per-channel protection scales: s = (s_act^alpha) / (s_w^(1 - alpha)).
pub fn awq_compute_channel_scales(s_act: &[f32], s_w: &[f32], alpha: f32) -> Vec<f32> {
    let eps = 1e-6f32;
    let d_in = s_act.len();
    let mut s = vec![0.0f32; d_in];
    let mut max_s = f32::MIN;
    let mut min_s = f32::MAX;

    for i in 0..d_in {
        let a = s_act[i].max(eps);
        let w = s_w[i].max(eps);
        let val = a.powf(alpha) / w.powf(1.0 - alpha);
        s[i] = val;
        if val > max_s {
            max_s = val;
        }
        if val < min_s {
            min_s = val;
        }
    }

    let norm = (max_s * min_s + eps).sqrt();
    for i in 0..d_in {
        s[i] /= norm;
    }
    s
}

/// Searches optimal AWQ alpha over candidate grid, minimizing output MSE.
pub fn awq_search_optimal_scale(
    x_calib: &Tensor,
    weight: &Tensor,
    alpha_grid: &[f32],
    qmin: f32,
    qmax: f32,
) -> Result<(Tensor, f32, Tensor)> {
    let (n_samples, d_in) = x_calib.dims2()?;
    let (_, d_out) = weight.dims2()?;
    let x_vec = x_calib.to_vec2::<f32>()?;
    let w_vec = weight.to_vec2::<f32>()?;

    // Compute s_act: mean(|x_calib|, dim=0)
    let mut s_act = vec![0.0f32; d_in];
    for col in 0..d_in {
        let mut sum = 0.0f32;
        for row in 0..n_samples {
            sum += x_vec[row][col].abs();
        }
        s_act[col] = sum / (n_samples as f32);
    }

    // Compute s_w: max(|weight|, dim=1)
    let mut s_w = vec![0.0f32; d_in];
    for row in 0..d_in {
        let mut max_val = 0.0f32;
        for col in 0..d_out {
            let v = w_vec[row][col].abs();
            if v > max_val {
                max_val = v;
            }
        }
        s_w[row] = max_val;
    }

    let y_ref = x_calib.matmul(weight)?;
    let mut best_loss = f32::INFINITY;
    let mut best_alpha = 0.5f32;
    let mut best_scale_vec = s_act.clone();
    let mut best_w_eff_flat = vec![0.0f32; d_in * d_out];

    for &alpha in alpha_grid {
        let s = awq_compute_channel_scales(&s_act, &s_w, alpha);

        // Scale weight rows
        let mut w_scaled_flat = vec![0.0f32; d_in * d_out];
        for i in 0..d_in {
            for j in 0..d_out {
                w_scaled_flat[i * d_out + j] = w_vec[i][j] * s[i];
            }
        }
        let w_scaled_t = Tensor::from_vec(w_scaled_flat, (d_in, d_out), weight.device())?;
        let (w_scaled_q, _) = rtn_quantize_per_column(&w_scaled_t, qmin, qmax)?;
        let q_vec = w_scaled_q.to_vec2::<f32>()?;

        // w_eff = w_scaled_q / s
        let mut w_eff_flat = vec![0.0f32; d_in * d_out];
        for i in 0..d_in {
            for j in 0..d_out {
                w_eff_flat[i * d_out + j] = q_vec[i][j] / s[i];
            }
        }
        let w_eff_t = Tensor::from_vec(w_eff_flat.clone(), (d_in, d_out), weight.device())?;
        let y_pred = x_calib.matmul(&w_eff_t)?;
        let diff = (&y_ref - &y_pred)?;
        let loss = diff.sqr()?.mean_all()?.to_scalar::<f32>()?;

        if loss < best_loss {
            best_loss = loss;
            best_alpha = alpha;
            best_scale_vec = s;
            best_w_eff_flat = w_eff_flat;
        }
    }

    let best_scale = Tensor::from_vec(best_scale_vec, (d_in,), weight.device())?;
    let best_w_eff = Tensor::from_vec(best_w_eff_flat, (d_in, d_out), weight.device())?;
    Ok((best_scale, best_alpha, best_w_eff))
}

/// Quantizes weight with a given AWQ channel scale vector.
pub fn awq_quantize_with_scale(
    weight: &Tensor,
    scale: &Tensor,
    qmin: f32,
    qmax: f32,
) -> Result<(Tensor, Tensor)> {
    let (d_in, d_out) = weight.dims2()?;
    let w_vec = weight.to_vec2::<f32>()?;
    let s_vec = scale.to_vec1::<f32>()?;

    let mut w_scaled_flat = vec![0.0f32; d_in * d_out];
    for i in 0..d_in {
        for j in 0..d_out {
            w_scaled_flat[i * d_out + j] = w_vec[i][j] * s_vec[i];
        }
    }
    let w_scaled_t = Tensor::from_vec(w_scaled_flat, (d_in, d_out), weight.device())?;
    let (w_scaled_q, col_scales) = rtn_quantize_per_column(&w_scaled_t, qmin, qmax)?;
    let q_vec = w_scaled_q.to_vec2::<f32>()?;

    let mut w_eff_flat = vec![0.0f32; d_in * d_out];
    for i in 0..d_in {
        for j in 0..d_out {
            w_eff_flat[i * d_out + j] = q_vec[i][j] / s_vec[i];
        }
    }
    let w_eff = Tensor::from_vec(w_eff_flat, (d_in, d_out), weight.device())?;
    Ok((w_eff, col_scales))
}
