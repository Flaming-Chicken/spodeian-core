//! Smoothquant and Outlier Migration W8A8
//! in candle/14_quantisation/03_llm_ptq_algorithms.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Computes SmoothQuant migration scale: s_j = (max |X_j|)^alpha / (max |W_j|)^(1 - alpha).
pub fn compute_smoothquant_scale(x_calib: &Tensor, weight: &Tensor, alpha: f32) -> Result<Tensor> {
    let eps = 1e-5f32;
    let (n_samples, d_in) = x_calib.dims2()?;
    let (_, d_out) = weight.dims2()?;
    let x_vec = x_calib.to_vec2::<f32>()?;
    let w_vec = weight.to_vec2::<f32>()?;

    let mut s = vec![0.0f32; d_in];
    for j in 0..d_in {
        let mut act_max = 0.0f32;
        for i in 0..n_samples {
            let v = x_vec[i][j].abs();
            if v > act_max {
                act_max = v;
            }
        }
        let mut weight_max = 0.0f32;
        for k in 0..d_out {
            let v = w_vec[j][k].abs();
            if v > weight_max {
                weight_max = v;
            }
        }
        let a = act_max.max(eps);
        let w = weight_max.max(eps);
        let s_j = (a.powf(alpha) / w.powf(1.0 - alpha)).max(eps);
        s[j] = s_j;
    }

    Tensor::from_vec(s, (d_in,), x_calib.device())
}

/// Per-token symmetric INT8 activation quantization.
pub fn quantize_per_token_int8(x: &Tensor) -> Result<(Tensor, Tensor)> {
    let (b, d) = x.dims2()?;
    let x_vec = x.to_vec2::<f32>()?;
    let mut deq_vec = vec![0.0f32; b * d];
    let mut scales = vec![0.0f32; b];

    for i in 0..b {
        let mut max_abs = 0.0f32;
        for j in 0..d {
            let v = x_vec[i][j].abs();
            if v > max_abs {
                max_abs = v;
            }
        }
        let s = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
        scales[i] = s;
        for j in 0..d {
            let orig = x_vec[i][j];
            let q = (orig / s).round().clamp(-128.0, 127.0);
            deq_vec[i * d + j] = q * s;
        }
    }

    let dequant = Tensor::from_vec(deq_vec, (b, d), x.device())?;
    let scale_tensor = Tensor::from_vec(scales, (b, 1), x.device())?;
    Ok((dequant, scale_tensor))
}

/// Per-column symmetric INT8 weight quantization.
pub fn quantize_per_channel_int8(w: &Tensor) -> Result<(Tensor, Tensor)> {
    let (d_in, d_out) = w.dims2()?;
    let w_vec = w.to_vec2::<f32>()?;
    let mut deq_vec = vec![0.0f32; d_in * d_out];
    let mut scales = vec![0.0f32; d_out];

    for j in 0..d_out {
        let mut max_abs = 0.0f32;
        for i in 0..d_in {
            let v = w_vec[i][j].abs();
            if v > max_abs {
                max_abs = v;
            }
        }
        let s = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
        scales[j] = s;
        for i in 0..d_in {
            let orig = w_vec[i][j];
            let q = (orig / s).round().clamp(-128.0, 127.0);
            deq_vec[i * d_out + j] = q * s;
        }
    }

    let dequant = Tensor::from_vec(deq_vec, (d_in, d_out), w.device())?;
    let scale_tensor = Tensor::from_vec(scales, (1, d_out), w.device())?;
    Ok((dequant, scale_tensor))
}

/// Standard W8A8 GEMM without smoothing.
pub fn w8a8_gemm_unfused(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let (deq_x, _) = quantize_per_token_int8(x)?;
    let (deq_w, _) = quantize_per_channel_int8(weight)?;
    let out = deq_x.matmul(&deq_w)?;
    match bias {
        Some(b) => out.broadcast_add(b),
        None => Ok(out),
    }
}

/// SmoothQuant W8A8 GEMM: scales activations and weights, then performs INT8 GEMM.
pub fn smoothquant_w8a8_gemm(
    x: &Tensor,
    weight: &Tensor,
    scale: &Tensor,
    bias: Option<&Tensor>,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (b, d_in) = x.dims2()?;
    let (_, d_out) = weight.dims2()?;
    let x_vec = x.to_vec2::<f32>()?;
    let w_vec = weight.to_vec2::<f32>()?;
    let s_vec = scale.to_vec1::<f32>()?;

    let mut x_smooth_flat = vec![0.0f32; b * d_in];
    for i in 0..b {
        for j in 0..d_in {
            x_smooth_flat[i * d_in + j] = x_vec[i][j] / s_vec[j];
        }
    }
    let x_smooth = Tensor::from_vec(x_smooth_flat, (b, d_in), x.device())?;

    let mut w_smooth_flat = vec![0.0f32; d_in * d_out];
    for i in 0..d_in {
        for j in 0..d_out {
            w_smooth_flat[i * d_out + j] = w_vec[i][j] * s_vec[i];
        }
    }
    let w_smooth = Tensor::from_vec(w_smooth_flat, (d_in, d_out), weight.device())?;

    let (deq_x, _) = quantize_per_token_int8(&x_smooth)?;
    let (deq_w, _) = quantize_per_channel_int8(&w_smooth)?;
    let out_raw = deq_x.matmul(&deq_w)?;
    let out = match bias {
        Some(b) => out_raw.broadcast_add(b)?,
        None => out_raw,
    };

    Ok((out, x_smooth, w_smooth))
}
