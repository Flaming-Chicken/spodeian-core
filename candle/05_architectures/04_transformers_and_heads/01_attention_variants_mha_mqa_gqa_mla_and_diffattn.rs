//! Attention Variants: MHA, MQA, GQA, MLA, and DiffAttn
//! in candle/05_architectures/04_transformers_and_heads.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Grouped-Query Attention (GQA).
pub fn grouped_query_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
) -> Result<(Tensor, Tensor)> {
    let q_dims = q.dims();
    let b_sz = q_dims[0];
    let h = q_dims[1];
    let _t_len = q_dims[2];
    let d = q_dims[3];

    let k_dims = k.dims();
    let g = k_dims[1];
    let s_len = k_dims[2];
    let group_size = h / g;

    // Broadcast k and v across group dimension: [B, G, 1, S, D] -> [B, G, group_size, S, D] -> [B, H, S, D]
    let k_exp = k
        .unsqueeze(2)?
        .broadcast_as((b_sz, g, group_size, s_len, d))?
        .contiguous()?
        .reshape((b_sz, h, s_len, d))?;
    let v_exp = v
        .unsqueeze(2)?
        .broadcast_as((b_sz, g, group_size, s_len, d))?
        .contiguous()?
        .reshape((b_sz, h, s_len, d))?;

    // Scaled dot-product: [B, H, T, D] @ [B, H, D, S] -> [B, H, T, S]
    let scores = (q.matmul(&k_exp.transpose(2, 3)?)? * scale)?;
    let attn_weights = candle_nn::ops::softmax(&scores, 3)?;

    let out = attn_weights.matmul(&v_exp)?;
    Ok((out, attn_weights))
}

/// Differential Attention (DiffAttn).
pub fn differential_attention(
    q1: &Tensor,
    k1: &Tensor,
    q2: &Tensor,
    k2: &Tensor,
    v: &Tensor,
    lambda_param: f64,
    scale: f64,
) -> Result<(Tensor, Tensor)> {
    let scores1 = (q1.matmul(&k1.transpose(2, 3)?)? * scale)?;
    let scores2 = (q2.matmul(&k2.transpose(2, 3)?)? * scale)?;

    let a1 = candle_nn::ops::softmax(&scores1, 3)?;
    let a2 = candle_nn::ops::softmax(&scores2, 3)?;

    let diff_weights = (a1 - (a2 * lambda_param)?)?;
    let out = diff_weights.matmul(v)?;

    Ok((out, diff_weights))
}
