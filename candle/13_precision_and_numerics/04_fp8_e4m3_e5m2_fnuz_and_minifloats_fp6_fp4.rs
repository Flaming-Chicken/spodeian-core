//! FP8 E4M3 E5M2 FNUZ and Minifloats FP6 FP4
//! in candle/13_precision_and_numerics.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Elementwise minifloat quantizer with subnormal support.
pub fn quantize_minifloat_f32(v: f32, _e_bits: i32, m_bits: i32, bias: i32, max_val: f32) -> f32 {
    if v == 0.0 {
        return 0.0;
    }
    let sign = if v < 0.0 { -1.0 } else { 1.0 };
    let abs_v = v.abs();
    let e = abs_v.log2().floor();
    let e_min_normal = 1.0 - (bias as f32);
    let step = if e < e_min_normal {
        (2.0f32).powf(e_min_normal - (m_bits as f32))
    } else {
        (2.0f32).powf(e - (m_bits as f32))
    };
    let q = (abs_v / step).round() * step;
    let q_clamped = q.clamp(0.0, max_val);
    sign * q_clamped
}

/// Quantize tensor to FP8 E4M3FN (1 sign, 4 exp bias 7, 3 mantissa, max 448.0).
pub fn fp8_e4m3fn_quantize(x: &Tensor, scale: f64) -> Result<Tensor> {
    let scale_f32 = scale as f32;
    let inv_scale = 1.0 / scale_f32;
    let vec = x.flatten_all()?.to_vec1::<f32>()?;
    let quantized: Vec<f32> = vec
        .into_iter()
        .map(|v| quantize_minifloat_f32(v * inv_scale, 4, 3, 7, 448.0) * scale_f32)
        .collect();
    Tensor::from_vec(quantized, x.shape(), x.device())
}

/// Quantize tensor to FP8 E5M2 (1 sign, 5 exp bias 15, 2 mantissa, max 57344.0).
pub fn fp8_e5m2_quantize(x: &Tensor, scale: f64) -> Result<Tensor> {
    let scale_f32 = scale as f32;
    let inv_scale = 1.0 / scale_f32;
    let vec = x.flatten_all()?.to_vec1::<f32>()?;
    let quantized: Vec<f32> = vec
        .into_iter()
        .map(|v| quantize_minifloat_f32(v * inv_scale, 5, 2, 15, 57344.0) * scale_f32)
        .collect();
    Tensor::from_vec(quantized, x.shape(), x.device())
}

/// Quantize tensor to OCP FP4 E2M1 (1 sign, 2 exp bias 1, 1 mantissa, max 6.0).
pub fn fp4_e2m1_quantize(x: &Tensor, scale: f64) -> Result<Tensor> {
    let scale_f32 = scale as f32;
    let inv_scale = 1.0 / scale_f32;
    let vec = x.flatten_all()?.to_vec1::<f32>()?;
    let quantized: Vec<f32> = vec
        .into_iter()
        .map(|v| quantize_minifloat_f32(v * inv_scale, 2, 1, 1, 6.0) * scale_f32)
        .collect();
    Tensor::from_vec(quantized, x.shape(), x.device())
}

/// Hopper-style FP8 GEMM with delayed scale computation.
pub fn delayed_scaled_fp8_gemm(
    a: &Tensor,
    b: &Tensor,
    a_scale: f64,
    b_scale: f64,
) -> Result<(Tensor, f64, f64)> {
    let a_q = fp8_e4m3fn_quantize(a, a_scale)?;
    let b_q = fp8_e4m3fn_quantize(b, b_scale)?;
    let out = a_q.matmul(&b_q)?;

    let a_max = a
        .flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .fold(0.0f32, |acc, v| acc.max(v.abs())) as f64;
    let b_max = b
        .flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .fold(0.0f32, |acc, v| acc.max(v.abs())) as f64;
    let next_a_scale = (a_max / 448.0).max(1e-6);
    let next_b_scale = (b_max / 448.0).max(1e-6);

    Ok((out, next_a_scale, next_b_scale))
}
