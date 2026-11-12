//! KV Cache Quantization: KIVI and KVQuant
//! in candle/14_quantisation/05_kv_attention_optim_and_benchmarks.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// KIVI Key Quantization: per-channel asymmetric quantization along sequence dimension (dim 0).
pub fn quantize_k_per_channel(k: &Tensor, bits: usize) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let qmax = ((1 << bits) - 1) as f32;
    let k_min = k.min_keepdim(0)?; // [1, d_k]
    let k_max = k.max_keepdim(0)?; // [1, d_k]
    let diff = (k_max - &k_min)?;
    let scale_raw = (diff / (qmax as f64))?;
    let scale = scale_raw.clamp(1e-7f32, f32::MAX)?;
    let zero = (k_min.neg()?.broadcast_div(&scale))?
        .clamp(0.0f32, qmax)?
        .round()?;
    let k_int = ((k.broadcast_div(&scale))?.broadcast_add(&zero))?
        .round()?
        .clamp(0.0f32, qmax)?;
    let k_deq = (k_int.broadcast_sub(&zero))?.broadcast_mul(&scale)?;
    Ok((k_deq, k_int, scale, zero))
}

/// KIVI Value Quantization: per-token asymmetric quantization along channel dimension (dim 1).
pub fn quantize_v_per_token(v: &Tensor, bits: usize) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let qmax = ((1 << bits) - 1) as f32;
    let v_min = v.min_keepdim(1)?; // [seq_len, 1]
    let v_max = v.max_keepdim(1)?; // [seq_len, 1]
    let diff = (v_max - &v_min)?;
    let scale_raw = (diff / (qmax as f64))?;
    let scale = scale_raw.clamp(1e-7f32, f32::MAX)?;
    let zero = (v_min.neg()?.broadcast_div(&scale))?
        .clamp(0.0f32, qmax)?
        .round()?;
    let v_int = ((v.broadcast_div(&scale))?.broadcast_add(&zero))?
        .round()?
        .clamp(0.0f32, qmax)?;
    let v_deq = (v_int.broadcast_sub(&zero))?.broadcast_mul(&scale)?;
    Ok((v_deq, v_int, scale, zero))
}

/// KIVI Attention Decoding: quantize past tokens while keeping recent window in FP32.
pub fn kivi_attention_decode(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    window_size: usize,
    bits: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let dims = k.shape().dims();
    let seq_len = dims[0];
    let d_k = dims[1];

    let (k_eff, v_eff) = if seq_len > window_size {
        let past_len = seq_len - window_size;
        let k_past = k.narrow(0, 0, past_len)?;
        let v_past = v.narrow(0, 0, past_len)?;
        let (k_past_deq, _, _, _) = quantize_k_per_channel(&k_past, bits)?;
        let (v_past_deq, _, _, _) = quantize_v_per_token(&v_past, bits)?;

        let k_recent = k.narrow(0, past_len, window_size)?;
        let v_recent = v.narrow(0, past_len, window_size)?;

        let k_eff = Tensor::cat(&[&k_past_deq, &k_recent], 0)?;
        let v_eff = Tensor::cat(&[&v_past_deq, &v_recent], 0)?;
        (k_eff, v_eff)
    } else {
        (k.clone(), v.clone())
    };

    let scores = (q.matmul(&k_eff.t()?)? * (1.0 / (d_k as f64).sqrt()))?;
    let probs = candle_nn::ops::softmax_last_dim(&scores)?;
    let out = probs.matmul(&v_eff)?;
    Ok((out, k_eff, v_eff))
}
