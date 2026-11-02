//! Quantized Attention: SageAttention INT8/FP8
//! in candle/14_quantisation/05_kv_attention_optim_and_benchmarks.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Key-mean smoothing: subtract sequence mean from Key matrix.
pub fn key_mean_smoothing(k: &Tensor) -> Result<(Tensor, Tensor)> {
    let k_mean = k.mean_keepdim(0)?;
    let k_smoothed = k.broadcast_sub(&k_mean)?;
    Ok((k_smoothed, k_mean))
}

/// Symmetric per-tensor INT8 quantization.
pub fn quantize_symmetric_int8(x: &Tensor, eps: f64) -> Result<(Tensor, Tensor, Tensor)> {
    let abs_max = x.abs()?.max_all()?.to_scalar::<f32>()?;
    let scale_val = (abs_max / 127.0f32).max(eps as f32);
    let scale = Tensor::new(&[scale_val], x.device())?;
    let scaled = x.broadcast_div(&scale)?;
    let x_int8 = scaled.round()?.clamp(-127.0f32, 127.0f32)?;
    let x_deq = x_int8.broadcast_mul(&scale)?;
    Ok((x_deq, x_int8, scale))
}

/// SageAttention forward pass: Key-mean smoothing -> INT8 QK^T -> Softmax -> INT8/FP8 PV.
pub fn sage_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
    let dims = q.shape().dims();
    let d_k = dims[1];

    let (k_smoothed, _) = key_mean_smoothing(k)?;
    let (_, q_int8, s_q) = quantize_symmetric_int8(q, 1e-7)?;
    let (_, k_int8, s_k) = quantize_symmetric_int8(&k_smoothed, 1e-7)?;

    let int_acc = q_int8.matmul(&k_int8.t()?)?;
    let scale_prod = (s_q.squeeze(0)?.to_scalar::<f32>()? * s_k.squeeze(0)?.to_scalar::<f32>()?)
        / (d_k as f32).sqrt();
    let scores = int_acc.affine(scale_prod as f64, 0.0)?;

    let probs = candle_nn::ops::softmax_last_dim(&scores)?;
    let (v_deq, _, _) = quantize_symmetric_int8(v, 1e-7)?;
    let out = probs.matmul(&v_deq)?;

    Ok((out, scores, probs, int_acc, k_smoothed))
}
