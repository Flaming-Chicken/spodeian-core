//! QAT STE LSQ LSQ+ PACT and DoReFa
//! in candle/14_quantisation/04_vector_lattice_qat_and_lowbit.

use candle_core::{DType, Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Straight-Through Estimator (STE) uniform quantization.
pub fn ste_quantize(w: &Tensor, scale: &Tensor, qmin: f64, qmax: f64) -> Result<(Tensor, Tensor)> {
    let s_val = scale.to_vec2::<f32>()?[0][0] as f64;
    let q = (w / s_val)?.round()?.clamp(qmin, qmax)?;
    let w_deq = (&q * s_val)?;

    let lower_bound = qmin * s_val;
    let upper_bound = qmax * s_val;
    let mask_ge = w.ge(lower_bound)?.to_dtype(DType::F32)?;
    let mask_le = w.le(upper_bound)?.to_dtype(DType::F32)?;
    let ste_mask = mask_ge.mul(&mask_le)?;

    Ok((w_deq, ste_mask))
}

/// Parameterized Activation Clipping (PACT) with learnable clipping threshold alpha.
pub fn pact_activation(x: &Tensor, alpha: &Tensor, nbits: usize) -> Result<(Tensor, Tensor)> {
    let a_val = alpha.to_vec0::<f32>()? as f64;
    let qmax = ((1 << nbits) - 1) as f64;
    let x_clipped = x.clamp(0.0, a_val)?;
    let step = a_val / qmax;
    let q = (&x_clipped / step)?.round()?;
    let x_deq = (&q * step)?;
    let grad_alpha_mask = x.ge(a_val)?.to_dtype(DType::F32)?;

    Ok((x_deq, grad_alpha_mask))
}

/// Learned Step Size Quantization (LSQ) with analytical gradient scaling.
pub fn lsq_quantize(w: &Tensor, scale: &Tensor, qmin: f64, qmax: f64) -> Result<(Tensor, Tensor)> {
    let s_val = scale.to_vec2::<f32>()?[0][0] as f64;
    let v = (w / s_val)?;
    let q = v.clamp(qmin, qmax)?.round()?;
    let w_deq = (&q * s_val)?;

    let numel = w.elem_count();
    let g = 1.0 / ((numel as f64 * qmax).sqrt());

    let v_lt = v.lt(qmin)?.to_dtype(DType::F32)?;
    let v_gt = v.gt(qmax)?.to_dtype(DType::F32)?;
    let v_in_range = (Tensor::ones_like(&v_lt)? - (&v_lt + &v_gt)?)?;
    let round_diff = (v.round()? - &v)?;

    let term_lt = (&v_lt * qmin)?;
    let term_gt = (&v_gt * qmax)?;
    let term_mid = v_in_range.mul(&round_diff)?;

    let d_w_bar_ds = ((&term_lt + &term_gt)? + &term_mid)?;
    let lsq_grad_scale = (&d_w_bar_ds * g)?;

    Ok((w_deq, lsq_grad_scale))
}

/// LSQ+ quantization with learnable scale and asymmetric offset beta.
pub fn lsq_plus_quantize(
    w: &Tensor,
    scale: &Tensor,
    beta: &Tensor,
    qmin: f64,
    qmax: f64,
) -> Result<Tensor> {
    let s_val = scale.to_vec2::<f32>()?[0][0] as f64;
    let v = (w.broadcast_sub(beta)? / s_val)?;
    let q = v.clamp(qmin, qmax)?.round()?;
    let w_scaled = (&q * s_val)?;
    w_scaled.broadcast_add(beta)
}

/// QAT end-to-end forward pass combining PACT activations and LSQ weights.
pub fn qat_forward(
    x: &Tensor,
    weight: &Tensor,
    scale_w: &Tensor,
    alpha_x: &Tensor,
    bias: Option<&Tensor>,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (x_q, _) = pact_activation(x, alpha_x, 4)?;
    let (w_q, _) = lsq_quantize(weight, scale_w, -8.0, 7.0)?;
    let mut y = x_q.matmul(&w_q)?;
    if let Some(b) = bias {
        y = y.broadcast_add(b)?;
    }
    Ok((y, x_q, w_q))
}
