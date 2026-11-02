//! Flash Attention, Online Softmax, and Flex Attention
//! in candle/05_architectures/04_transformers_and_heads.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Tiled FlashAttention algorithm with Online Softmax accumulation.
pub fn online_softmax_flash_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    block_size: usize,
    scale: f64,
) -> Result<(Tensor, Tensor, Tensor)> {
    let q_dims = q.dims();
    let b_sz = q_dims[0];
    let t_len = q_dims[1];
    let d = q_dims[2];

    let s_len = k.dims()[1];
    let num_blocks = (s_len + block_size - 1) / block_size;

    let mut out = Tensor::zeros((b_sz, t_len, d), candle_core::DType::F32, q.device())?;
    let mut m = (Tensor::ones((b_sz, t_len, 1), candle_core::DType::F32, q.device())? * (-1e9))?;
    let mut l = Tensor::zeros((b_sz, t_len, 1), candle_core::DType::F32, q.device())?;

    for j in 0..num_blocks {
        let start = j * block_size;
        let len = (s_len - start).min(block_size);

        let k_block = k.narrow(1, start, len)?;
        let v_block = v.narrow(1, start, len)?;

        // [B, T, D] @ [B, D, len] -> [B, T, len]
        let scores_j = (q.matmul(&k_block.transpose(1, 2)?)? * scale)?;
        let block_max = scores_j.max_keepdim(2)?; // [B, T, 1]
        let m_new = m.maximum(&block_max)?;

        let alpha = m.sub(&m_new)?.exp()?; // [B, T, 1]
        let p_j = scores_j.broadcast_sub(&m_new)?.exp()?; // [B, T, len]

        let l_new = (alpha.mul(&l)? + p_j.sum_keepdim(2)?)?; // [B, T, 1]
        let pv = p_j.matmul(&v_block)?; // [B, T, D]

        let out_new =
            (alpha.broadcast_mul(&l)?.broadcast_mul(&out)? + pv)?.broadcast_div(&l_new)?;
        out = out_new;
        m = m_new;
        l = l_new;
    }

    Ok((out, m, l))
}
