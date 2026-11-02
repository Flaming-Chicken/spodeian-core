//! BitNet b1.58 Native Ternary Transformers
//! in candle/14_quantisation/04_vector_lattice_qat_and_lowbit.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// BitNet b1.58 weight quantization: scales by mean(|W|) and clips to {-1, 0, +1}.
pub fn weight_quant_bitnet(w: &Tensor, eps: f64) -> Result<(Tensor, Tensor, Tensor)> {
    let abs_w = w.abs()?;
    let mean_val = abs_w.mean_all()?.to_scalar::<f32>()?;
    let beta = Tensor::new(&[[mean_val]], w.device())?;
    let scale_val = if (mean_val as f64) < eps {
        eps as f32
    } else {
        mean_val
    };
    let scale = Tensor::new(&[[scale_val]], w.device())?;
    let scaled_w = w.broadcast_div(&scale)?;
    let w_ternary = scaled_w.round()?.clamp(-1.0f32, 1.0f32)?;
    let w_deq = w_ternary.broadcast_mul(&scale)?;
    Ok((w_deq, w_ternary, beta))
}

/// BitNet b1.58 per-token activation quantization to 8-bit integers.
pub fn activation_quant_bitnet(x: &Tensor, q_b: f64, eps: f64) -> Result<(Tensor, Tensor, Tensor)> {
    let gamma = x.abs()?.max_keepdim(1)?;
    let scale_raw =
        (gamma.to_dtype(candle_core::DType::F64)? / q_b)?.to_dtype(candle_core::DType::F32)?;
    let scale = scale_raw.clamp(eps as f32, f32::MAX)?;
    let scaled_x = x.broadcast_div(&scale)?;
    let x_int8 = scaled_x.round()?.clamp(-q_b as f32, q_b as f32)?;
    let x_deq = x_int8.broadcast_mul(&scale)?;
    Ok((x_deq, x_int8, gamma))
}

/// Root Mean Square Layer Normalization (SubLN / RMSNorm).
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let variance = x.sqr()?.mean_keepdim(1)?;
    let var_eps = variance.affine(1.0, eps)?;
    let rsqrt = var_eps.sqrt()?.recip()?;
    let norm = x.broadcast_mul(&rsqrt)?;
    norm.broadcast_mul(weight)
}

/// BitLinear layer forward pass: SubLN -> INT8 act quant -> Ternary weight quant -> MatMul.
pub fn bitlinear_forward(
    x: &Tensor,
    weight: &Tensor,
    rms_weight: &Tensor,
    bias: Option<&Tensor>,
) -> Result<(Tensor, Tensor, Tensor)> {
    let x_norm = rms_norm(x, rms_weight, 1e-5)?;
    let (x_q, _, _) = activation_quant_bitnet(&x_norm, 127.0, 1e-5)?;
    let (w_q, _, _) = weight_quant_bitnet(weight, 1e-5)?;
    let mut y = x_q.matmul(&w_q)?;
    if let Some(b) = bias {
        y = y.broadcast_add(b)?;
    }
    Ok((y, x_q, w_q))
}
