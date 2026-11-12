//! Sparse, Sliding Window, and Sink Attention
//! in candle/05_architectures/04_transformers_and_heads.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// StreamingLLM Attention with Attention Sinks and Sliding Window.
pub fn streaming_sink_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    num_sinks: usize,
    window_size: usize,
    scale: f64,
) -> Result<(Tensor, Tensor)> {
    let t_len = q.dims()[1];
    let s_len = k.dims()[1];

    let scores = (q.matmul(&k.transpose(1, 2)?)? * scale)?;

    let mut mask_vals = vec![0.0f32; t_len * s_len];
    for i in 0..t_len {
        for j in 0..s_len {
            let is_sink = j < num_sinks;
            let is_window = i >= j && (i - j) < window_size;
            let valid = (is_sink || is_window) && j <= i;
            mask_vals[i * s_len + j] = if valid { 0.0 } else { -1e9 };
        }
    }
    let mask_t = Tensor::from_vec(mask_vals, (t_len, s_len), q.device())?;
    let masked_scores = scores.broadcast_add(&mask_t.unsqueeze(0)?)?;

    let attn_weights = candle_nn::ops::softmax(&masked_scores, 2)?;
    let out = attn_weights.matmul(v)?;

    Ok((out, attn_weights))
}
