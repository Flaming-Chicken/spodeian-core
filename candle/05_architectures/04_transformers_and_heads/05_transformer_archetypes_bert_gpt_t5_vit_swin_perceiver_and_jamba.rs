//! Transformer Archetypes: BERT, GPT, T5, ViT, Swin, Perceiver, and Jamba
//! in candle/05_architectures/04_transformers_and_heads.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Helper to multiply 3D tensor [B, T, D_in] with 2D weight [D_in, D_out] -> [B, T, D_out].
pub fn matmul_3d_2d(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    let (d0, d1, d2) = x.dims3()?;
    let x_2d = x.contiguous()?.reshape((d0 * d1, d2))?;
    let out_2d = x_2d.matmul(w)?;
    let out_dim = w.dim(1)?;
    out_2d.reshape((d0, d1, out_dim))
}

/// Perceiver cross-attention bottleneck.
pub fn perceiver_cross_attention(
    latents: &Tensor,
    inputs: &Tensor,
    w_q: &Tensor,
    w_k: &Tensor,
    w_v: &Tensor,
    w_o: &Tensor,
    scale: f64,
) -> Result<(Tensor, Tensor)> {
    let q = matmul_3d_2d(latents, w_q)?; // [B, N, D_head]
    let k = matmul_3d_2d(inputs, w_k)?; // [B, M, D_head]
    let v = matmul_3d_2d(inputs, w_v)?; // [B, M, D_head]

    let scores = (q.matmul(&k.transpose(1, 2)?)? * scale)?;
    let attn_weights = candle_nn::ops::softmax(&scores, 2)?;

    let ctx = attn_weights.matmul(&v)?;
    let out = matmul_3d_2d(&ctx, w_o)?;

    Ok((out, attn_weights))
}

/// Causal autoregressive decoder mask (GPT archetype).
pub fn causal_decoder_mask(scores: &Tensor) -> Result<Tensor> {
    let t = scores.dim(candle_core::D::Minus1)?;
    let mut mask_vals = vec![0.0f32; t * t];
    for i in 0..t {
        for j in 0..t {
            mask_vals[i * t + j] = if j <= i { 0.0 } else { -1e9 };
        }
    }
    let mask_t = Tensor::from_vec(mask_vals, (t, t), scores.device())?;
    scores.broadcast_add(&mask_t.unsqueeze(0)?)
}
