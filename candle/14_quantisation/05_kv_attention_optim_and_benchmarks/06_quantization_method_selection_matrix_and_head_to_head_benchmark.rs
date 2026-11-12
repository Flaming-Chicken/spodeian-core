//! Quantization Method Selection Matrix and Head to Head Benchmark
//! in candle/14_quantisation/05_kv_attention_optim_and_benchmarks.

use candle_core::{Result, Tensor};

pub const NF4_LEVELS: [f32; 16] = [
    -1.0,
    -0.6961928,
    -0.52507305,
    -0.3949175,
    -0.28444138,
    -0.18477343,
    -0.091050036,
    0.0,
    0.0795803,
    0.1609302,
    0.2461123,
    0.33791524,
    0.44070983,
    0.562617,
    0.72295684,
    1.0,
];

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Round-to-Nearest uniform symmetric per-row quantization.
pub fn rtn_quantize(w: &Tensor, bits: usize) -> Result<Tensor> {
    let levels = ((1usize << (bits - 1)) - 1) as f64;
    let max_abs = w.abs()?.max_keepdim(1)?;
    let scale = (max_abs / levels)?;
    let scale_eps = scale.affine(1.0, 1e-8)?;
    let q = w.broadcast_div(&scale_eps)?.round()?;
    let min_b = Tensor::new(-levels as f32, w.device())?.broadcast_as(w.shape())?;
    let max_b = Tensor::new(levels as f32, w.device())?.broadcast_as(w.shape())?;
    let q_c = q.clamp(&min_b, &max_b)?;
    q_c.broadcast_mul(&scale)
}

/// Activation-Aware Weight Quantization (AWQ) with salient column protection.
pub fn awq_quantize(w: &Tensor, x: &Tensor, bits: usize) -> Result<Tensor> {
    let levels = ((1usize << (bits - 1)) - 1) as f64;
    let act_scale = x.abs()?.mean_keepdim(0)?;
    let min_scale = Tensor::new(1e-3f32, w.device())?.broadcast_as(act_scale.shape())?;
    let max_scale = Tensor::new(1e6f32, w.device())?.broadcast_as(act_scale.shape())?;
    let s_awq = act_scale.sqrt()?.clamp(&min_scale, &max_scale)?;
    let w_scaled = w.broadcast_mul(&s_awq)?;

    let max_w = w_scaled.abs()?.max_keepdim(1)?;
    let s_w = (max_w / levels)?;
    let s_w_eps = s_w.affine(1.0, 1e-8)?;
    let q = w_scaled.broadcast_div(&s_w_eps)?.round()?;
    let min_b = Tensor::new(-levels as f32, w.device())?.broadcast_as(w.shape())?;
    let max_b = Tensor::new(levels as f32, w.device())?.broadcast_as(w.shape())?;
    let q_c = q.clamp(&min_b, &max_b)?;
    q_c.broadcast_mul(&s_w)?.broadcast_div(&s_awq)
}

/// BitNet b1.58 native ternary {-scale, 0, +scale} quantization.
pub fn bitnet_quantize(w: &Tensor) -> Result<Tensor> {
    let scale = w.abs()?.mean_all()?.to_scalar::<f32>()? as f64;
    let scale_eps = scale + 1e-8;
    let q = (w / scale_eps)?.round()?;
    let min_b = Tensor::new(-1.0f32, w.device())?.broadcast_as(w.shape())?;
    let max_b = Tensor::new(1.0f32, w.device())?.broadcast_as(w.shape())?;
    let q_c = q.clamp(&min_b, &max_b)?;
    q_c * scale
}

/// NormalFloat 4-bit (NF4) quantile codebook quantization.
pub fn nf4_quantize(w: &Tensor) -> Result<Tensor> {
    let scale = w.abs()?.max_keepdim(1)?;
    let scale_eps = scale.affine(1.0, 1e-8)?;
    let norm_w = w.broadcast_div(&scale_eps)?;

    let (rows, cols) = norm_w.dims2()?;
    let data = norm_w.to_vec2::<f32>()?;
    let mut quantized_data = Vec::with_capacity(rows * cols);

    for r in 0..rows {
        for c in 0..cols {
            let val = data[r][c];
            let mut best_level = NF4_LEVELS[0];
            let mut min_dist = (val - best_level).abs();
            for &lvl in &NF4_LEVELS[1..] {
                let dist = (val - lvl).abs();
                if dist < min_dist {
                    min_dist = dist;
                    best_level = lvl;
                }
            }
            quantized_data.push(best_level);
        }
    }

    let q_tensor = Tensor::from_vec(quantized_data, (rows, cols), w.device())?;
    q_tensor.broadcast_mul(&scale)
}

/// Computes Signal-to-Quantization-Noise Ratio (SQNR) in dB.
pub fn compute_sqnr(signal: &Tensor, noise: &Tensor) -> Result<f32> {
    let sig_pwr = signal.sqr()?.mean_all()?.to_scalar::<f32>()? as f64;
    let noise_pwr = noise.sqr()?.mean_all()?.to_scalar::<f32>()? as f64;
    Ok((10.0 * (sig_pwr / (noise_pwr + 1e-12)).log10()) as f32)
}
