//! Logarithmic PoT APoT μ-Law and Posit Quantization
//! in candle/14_quantisation/01_foundations_formats_and_observers.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Power-of-Two (PoT) multiplier-free quantization.
pub fn pot_quantize(x: &Tensor, num_bits: usize) -> Result<Tensor> {
    let vec = x.flatten_all()?.to_vec1::<f32>()?;
    let max_p = ((1usize << (num_bits - 1)) - 2) as f32;
    let threshold = (2.0f32).powf(-max_p - 1.0);

    let out: Vec<f32> = vec
        .into_iter()
        .map(|v| {
            let abs_v = v.abs();
            if abs_v < threshold {
                0.0f32
            } else {
                let sign = if v < 0.0 { -1.0f32 } else { 1.0f32 };
                let p = (-abs_v.log2()).round().clamp(0.0, max_p);
                sign * (2.0f32).powf(-p)
            }
        })
        .collect();

    Tensor::from_vec(out, x.shape(), x.device())
}

/// Additive Power-of-Two (APoT) 4-bit quantization with 8 non-negative levels.
pub fn apot_quantize(x: &Tensor) -> Result<Tensor> {
    let levels = [0.0f32, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0];
    let vec = x.flatten_all()?.to_vec1::<f32>()?;

    let out: Vec<f32> = vec
        .into_iter()
        .map(|v| {
            let sign = if v < 0.0 { -1.0f32 } else { 1.0f32 };
            let abs_v = v.abs().clamp(0.0, 1.0);
            let mut best_level = levels[0];
            let mut min_diff = (abs_v - levels[0]).abs();

            for &lvl in &levels[1..] {
                let diff = (abs_v - lvl).abs();
                if diff < min_diff {
                    min_diff = diff;
                    best_level = lvl;
                }
            }
            sign * best_level
        })
        .collect();

    Tensor::from_vec(out, x.shape(), x.device())
}

/// ITU-T G.711 / WaveNet μ-law non-linear companding and expansion.
pub fn mu_law_compand_expand(x: &Tensor, mu: f64, num_bits: usize) -> Result<Tensor> {
    let mu_f32 = mu as f32;
    let ln_1_mu = (1.0f32 + mu_f32).ln();
    let qmax = ((1usize << num_bits) - 1) as f32;
    let vec = x.flatten_all()?.to_vec1::<f32>()?;

    let out: Vec<f32> = vec
        .into_iter()
        .map(|v| {
            let x_clamped = v.clamp(-1.0, 1.0);
            let sign = if x_clamped < 0.0 { -1.0f32 } else { 1.0f32 };
            // Compand
            let y = sign * (1.0f32 + mu_f32 * x_clamped.abs()).ln() / ln_1_mu;
            // Uniform quantize
            let y_norm = (y + 1.0) * 0.5;
            let y_q = (y_norm * qmax).round() / qmax;
            let y_quant = y_q * 2.0 - 1.0;
            // Expand
            let sign_q = if y_quant < 0.0 { -1.0f32 } else { 1.0f32 };
            sign_q * ((1.0f32 + mu_f32).powf(y_quant.abs()) - 1.0) / mu_f32
        })
        .collect();

    Tensor::from_vec(out, x.shape(), x.device())
}

/// Base-sqrt(2) logarithmic quantization.
pub fn base_sqrt2_log_quantize(x: &Tensor) -> Result<Tensor> {
    let threshold = (2.0f32).powf(-7.5);
    let vec = x.flatten_all()?.to_vec1::<f32>()?;

    let out: Vec<f32> = vec
        .into_iter()
        .map(|v| {
            let abs_v = v.abs();
            if abs_v < threshold {
                0.0f32
            } else {
                let sign = if v < 0.0 { -1.0f32 } else { 1.0f32 };
                let k = (2.0f32 * abs_v.log2()).round().clamp(-14.0, 0.0);
                sign * (2.0f32).powf(k * 0.5)
            }
        })
        .collect();

    Tensor::from_vec(out, x.shape(), x.device())
}

/// 4-bit Posit tapered quantization codebook.
pub fn posit4_quantize(x: &Tensor) -> Result<Tensor> {
    let levels = [0.0f32, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0];
    let vec = x.flatten_all()?.to_vec1::<f32>()?;

    let out: Vec<f32> = vec
        .into_iter()
        .map(|v| {
            let sign = if v < 0.0 { -1.0f32 } else { 1.0f32 };
            let abs_v = v.abs().clamp(0.0, 4.0);
            let mut best_level = levels[0];
            let mut min_diff = (abs_v - levels[0]).abs();

            for &lvl in &levels[1..] {
                let diff = (abs_v - lvl).abs();
                if diff < min_diff {
                    min_diff = diff;
                    best_level = lvl;
                }
            }
            sign * best_level
        })
        .collect();

    Tensor::from_vec(out, x.shape(), x.device())
}
