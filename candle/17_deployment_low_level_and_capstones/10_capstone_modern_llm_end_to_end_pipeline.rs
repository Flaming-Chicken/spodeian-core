//! Capstone Modern LLM End to End Pipeline
//! in candle/17_deployment_low_level_and_capstones.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Helper for batched 3D tensor multiplied by 2D weight matrix: [B, S, D] x [D, H] -> [B, S, H].
pub fn matmul_3d_2d(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    let (b, s, d) = x.dims3()?;
    let h = w.dim(1)?;
    let flat_x = x.reshape((b * s, d))?;
    let flat_y = flat_x.matmul(w)?;
    flat_y.reshape((b, s, h))
}

/// Root Mean Square Normalization.
pub fn rmsnorm(x: &Tensor, gamma: &Tensor, eps: f64) -> Result<Tensor> {
    let sq = x.sqr()?;
    let mean_sq = sq.mean_keepdim(candle_core::D::Minus1)?;
    let denom = mean_sq.affine(1.0, eps)?.sqrt()?;
    let norm = x.broadcast_div(&denom)?;
    norm.broadcast_mul(gamma)
}

/// Causal self-attention layer with causal upper-triangular masking.
pub fn causal_attention_block(
    x: &Tensor,
    wq: &Tensor,
    wk: &Tensor,
    wv: &Tensor,
    wo: &Tensor,
) -> Result<Tensor> {
    let (_b, s, d) = x.dims3()?;
    let q = matmul_3d_2d(x, wq)?;
    let k = matmul_3d_2d(x, wk)?;
    let v = matmul_3d_2d(x, wv)?;

    let k_t = k.transpose(1, 2)?;
    let scores = q.matmul(&k_t)?;
    let scale = 1.0 / (d as f64).sqrt();
    let scaled_scores = scores.affine(scale, 0.0)?;

    let mut mask_vec = vec![0.0f32; s * s];
    for i in 0..s {
        for j in 0..s {
            if j > i {
                mask_vec[i * s + j] = -10000.0;
            }
        }
    }
    let mask = Tensor::from_vec(mask_vec, (s, s), x.device())?;
    let masked_scores = scaled_scores.broadcast_add(&mask)?;

    let attn = candle_nn::ops::softmax(&masked_scores, candle_core::D::Minus1)?;
    let context = attn.matmul(&v)?;
    matmul_3d_2d(&context, wo)
}

/// SwiGLU feed-forward network block.
pub fn swiglu_ffn_block(
    x: &Tensor,
    w_gate: &Tensor,
    w_up: &Tensor,
    w_down: &Tensor,
) -> Result<Tensor> {
    let gate = candle_nn::ops::silu(&(matmul_3d_2d(x, w_gate)?))?;
    let up = matmul_3d_2d(x, w_up)?;
    let gated = gate.mul(&up)?;
    matmul_3d_2d(&gated, w_down)
}

/// End-to-end modern LLM forward pipeline.
pub fn modern_llm_pipeline_forward(
    x: &Tensor,
    gamma1: &Tensor,
    wq: &Tensor,
    wk: &Tensor,
    wv: &Tensor,
    wo: &Tensor,
    gamma2: &Tensor,
    w_gate: &Tensor,
    w_up: &Tensor,
    w_down: &Tensor,
    gamma_final: &Tensor,
    w_head: &Tensor,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let normed_1 = rmsnorm(x, gamma1, 1e-5)?;
    let attn = causal_attention_block(&normed_1, wq, wk, wv, wo)?;
    let h1 = (x + &attn)?;

    let normed_2 = rmsnorm(&h1, gamma2, 1e-5)?;
    let ffn = swiglu_ffn_block(&normed_2, w_gate, w_up, w_down)?;
    let h_out = (&h1 + &ffn)?;

    let h_final = rmsnorm(&h_out, gamma_final, 1e-5)?;
    let logits = matmul_3d_2d(&h_final, w_head)?;

    Ok((logits, h_out, attn, ffn))
}
