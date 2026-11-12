//! OmniQuant and HQQ Optimization-Based PTQ
//! in candle/14_quantisation/03_llm_ptq_algorithms.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

fn sigmoid(v: f32) -> f32 {
    1.0f32 / (1.0f32 + (-v).exp())
}

/// Learnable Weight Clipping (LWC) asymmetric quantization.
pub fn lwc_quantize(
    w: &Tensor,
    h_l: &Tensor,
    h_u: &Tensor,
    qmin: f32,
    qmax: f32,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (d_in, d_out) = w.dims2()?;
    let w_vec = w.to_vec2::<f32>()?;
    let hl_vec = h_l.to_vec2::<f32>()?[0].clone();
    let hu_vec = h_u.to_vec2::<f32>()?[0].clone();

    let mut w_deq_flat = vec![0.0f32; d_in * d_out];
    let mut scales = vec![0.0f32; d_out];
    let mut lowers = vec![0.0f32; d_out];

    for j in 0..d_out {
        let mut min_val = f32::MAX;
        let mut max_val = f32::MIN;
        for i in 0..d_in {
            let v = w_vec[i][j];
            if v < min_val {
                min_val = v;
            }
            if v > max_val {
                max_val = v;
            }
        }
        let lower = min_val * sigmoid(hl_vec[j]);
        let upper = max_val * sigmoid(hu_vec[j]);
        let scale = ((upper - lower) / qmax).max(1e-5);
        scales[j] = scale;
        lowers[j] = lower;

        for i in 0..d_in {
            let orig = w_vec[i][j];
            let q = ((orig - lower) / scale).round().clamp(qmin, qmax);
            w_deq_flat[i * d_out + j] = q * scale + lower;
        }
    }

    let w_deq = Tensor::from_vec(w_deq_flat, (d_in, d_out), w.device())?;
    let scale_tensor = Tensor::from_vec(scales, (1, d_out), w.device())?;
    let lower_tensor = Tensor::from_vec(lowers, (1, d_out), w.device())?;
    Ok((w_deq, scale_tensor, lower_tensor))
}

/// OmniQuant forward pass combining Learnable Equivalent Transformation (LET) & LWC.
pub fn omniquant_forward(
    x: &Tensor,
    weight: &Tensor,
    gamma: &Tensor,
    h_l: &Tensor,
    h_u: &Tensor,
    bias: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    let (b, d_in) = x.dims2()?;
    let (_, d_out) = weight.dims2()?;
    let x_vec = x.to_vec2::<f32>()?;
    let w_vec = weight.to_vec2::<f32>()?;
    let g_vec = gamma.to_vec1::<f32>()?;

    let s: Vec<f32> = g_vec.iter().map(|&g| g.exp()).collect();

    let mut x_scaled_flat = vec![0.0f32; b * d_in];
    for i in 0..b {
        for j in 0..d_in {
            x_scaled_flat[i * d_in + j] = x_vec[i][j] / s[j];
        }
    }
    let x_scaled = Tensor::from_vec(x_scaled_flat, (b, d_in), x.device())?;

    let mut w_scaled_flat = vec![0.0f32; d_in * d_out];
    for i in 0..d_in {
        for j in 0..d_out {
            w_scaled_flat[i * d_out + j] = w_vec[i][j] * s[i];
        }
    }
    let w_scaled = Tensor::from_vec(w_scaled_flat, (d_in, d_out), weight.device())?;

    let (w_deq, _, _) = lwc_quantize(&w_scaled, h_l, h_u, 0.0, 15.0)?;
    let mut out = x_scaled.matmul(&w_deq)?;
    if let Some(bi) = bias {
        out = out.broadcast_add(bi)?;
    }
    Ok((out, w_deq))
}

/// Half-Quadratic Quantization (HQQ) closed-form optimal scale and zero-point derivation.
pub fn hqq_proximal_quantize(w: &Tensor, nbits: usize) -> Result<(Tensor, Tensor, Tensor)> {
    let qmax = ((1 << nbits) - 1) as f32;
    let (d_in, d_out) = w.dims2()?;
    let w_vec = w.to_vec2::<f32>()?;

    let mut w_deq_flat = vec![0.0f32; d_in * d_out];
    let mut scales = vec![0.0f32; d_out];
    let mut zeros = vec![0.0f32; d_out];

    for j in 0..d_out {
        let mut min_val = f32::MAX;
        let mut max_val = f32::MIN;
        for i in 0..d_in {
            let v = w_vec[i][j];
            if v < min_val {
                min_val = v;
            }
            if v > max_val {
                max_val = v;
            }
        }
        let scale = ((max_val - min_val) / qmax).max(1e-5);
        let zero = -min_val / scale;
        scales[j] = scale;
        zeros[j] = zero;

        for i in 0..d_in {
            let orig = w_vec[i][j];
            let q = (orig / scale + zero).round().clamp(0.0, qmax);
            w_deq_flat[i * d_out + j] = (q - zero) * scale;
        }
    }

    let w_deq = Tensor::from_vec(w_deq_flat, (d_in, d_out), w.device())?;
    let scale_tensor = Tensor::from_vec(scales, (1, d_out), w.device())?;
    let zero_tensor = Tensor::from_vec(zeros, (1, d_out), w.device())?;
    Ok((w_deq, scale_tensor, zero_tensor))
}
