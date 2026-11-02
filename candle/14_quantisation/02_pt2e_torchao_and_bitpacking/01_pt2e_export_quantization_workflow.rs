//! PT2E Export Quantization Workflow
//! in candle/14_quantisation/02_pt2e_torchao_and_bitpacking.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Algebraic Conv2d + BatchNorm2d fusion: W_f = W * (gamma / sqrt(var + eps)).
pub fn conv2d_bn_fold(
    w_conv: &Tensor,
    b_conv: &Tensor,
    bn_mean: &Tensor,
    bn_var: &Tensor,
    bn_gamma: &Tensor,
    bn_beta: &Tensor,
    eps: f64,
) -> Result<(Tensor, Tensor)> {
    let denom = (bn_var.affine(1.0, eps)?).sqrt()?;
    let scale = bn_gamma.div(&denom)?;

    let c_out = w_conv.dim(0)?;
    let scale_4d = scale.reshape((c_out, 1, 1, 1))?;
    let w_folded = w_conv.broadcast_mul(&scale_4d)?;

    let b_sub_mean = b_conv.sub(bn_mean)?;
    let b_scaled = b_sub_mean.mul(&scale)?;
    let b_folded = b_scaled.add(bn_beta)?;

    Ok((w_folded, b_folded))
}

/// Algebraic Linear + BatchNorm1d fusion: W_f = W * (gamma / sqrt(var + eps)).
pub fn linear_bn_fold(
    w_linear: &Tensor,
    b_linear: &Tensor,
    bn_mean: &Tensor,
    bn_var: &Tensor,
    bn_gamma: &Tensor,
    bn_beta: &Tensor,
    eps: f64,
) -> Result<(Tensor, Tensor)> {
    let denom = (bn_var.affine(1.0, eps)?).sqrt()?;
    let scale = bn_gamma.div(&denom)?;

    let d_out = w_linear.dim(1)?;
    let scale_2d = scale.reshape((1, d_out))?;
    let w_folded = w_linear.broadcast_mul(&scale_2d)?;

    let b_sub_mean = b_linear.sub(bn_mean)?;
    let b_scaled = b_sub_mean.mul(&scale)?;
    let b_folded = b_scaled.add(bn_beta)?;

    Ok((w_folded, b_folded))
}

/// Simulated INT8 Quantize-Dequantize node.
pub fn quantize_symmetric_int8(x: &Tensor, scale: f32, qmin: f32, qmax: f32) -> Result<Tensor> {
    if scale == 0.0 {
        return Tensor::zeros(x.shape(), x.dtype(), x.device());
    }
    let flat = x.flatten_all()?.to_vec1::<f32>()?;
    let inv_scale = 1.0f32 / scale;
    let out: Vec<f32> = flat
        .into_iter()
        .map(|v| (v * inv_scale).round().clamp(qmin, qmax) * scale)
        .collect();
    Tensor::from_vec(out, x.shape(), x.device())
}

/// Conv2d helper with bias broadcasting.
fn conv2d_with_bias(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let out = x.conv2d(weight, 0, 1, 1, 1)?;
    let c_out = bias.dim(0)?;
    let b_4d = bias.reshape((1, c_out, 1, 1))?;
    out.broadcast_add(&b_4d)
}

/// Low-level Q/DQ Conv2d node following PT2E lowering graph.
pub fn pt2e_qdq_conv2d(
    x: &Tensor,
    w_folded: &Tensor,
    b_folded: &Tensor,
    s_x: f32,
    s_w: f32,
) -> Result<Tensor> {
    let x_q = quantize_symmetric_int8(x, s_x, -128.0, 127.0)?;
    let w_q = quantize_symmetric_int8(w_folded, s_w, -128.0, 127.0)?;
    conv2d_with_bias(&x_q, &w_q, b_folded)
}

/// Low-level Q/DQ Linear node following PT2E lowering graph.
pub fn pt2e_qdq_linear(
    x: &Tensor,
    w_folded: &Tensor,
    b_folded: &Tensor,
    s_x: f32,
    s_w: f32,
) -> Result<Tensor> {
    let x_q = quantize_symmetric_int8(x, s_x, -128.0, 127.0)?;
    let w_q = quantize_symmetric_int8(w_folded, s_w, -128.0, 127.0)?;
    x_q.matmul(&w_q)?.broadcast_add(b_folded)
}

/// Pathological anti-pattern: quantizing Conv weights before unfused BatchNorm.
pub fn unfused_conv_bn_quantize(
    x: &Tensor,
    w_conv: &Tensor,
    b_conv: &Tensor,
    bn_mean: &Tensor,
    bn_var: &Tensor,
    bn_gamma: &Tensor,
    bn_beta: &Tensor,
    s_w: f32,
    eps: f64,
) -> Result<Tensor> {
    let w_q = quantize_symmetric_int8(w_conv, s_w, -128.0, 127.0)?;
    let conv_out = conv2d_with_bias(x, &w_q, b_conv)?;

    let denom = (bn_var.affine(1.0, eps)?).sqrt()?;
    let scale = bn_gamma.div(&denom)?;

    let c_out = bn_mean.dim(0)?;
    let mean_4d = bn_mean.reshape((1, c_out, 1, 1))?;
    let scale_4d = scale.reshape((1, c_out, 1, 1))?;
    let beta_4d = bn_beta.reshape((1, c_out, 1, 1))?;

    let sub = conv_out.broadcast_sub(&mean_4d)?;
    let mul = sub.broadcast_mul(&scale_4d)?;
    mul.broadcast_add(&beta_4d)
}
