//! Transformer Heads: Induction, OV/QK Circuits, MTP, and Registers
//! in candle/05_architectures/04_transformers_and_heads.

use candle_core::{IndexOp, Result, Tensor};

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

/// Evaluates decoupled QK (attention pattern) and OV (value copy) circuits.
pub fn ov_qk_circuits(
    x_q: &Tensor,
    x_k: &Tensor,
    x_v: &Tensor,
    w_q: &Tensor,
    w_k: &Tensor,
    w_v: &Tensor,
    w_o: &Tensor,
    scale: f64,
) -> Result<(Tensor, Tensor)> {
    let q = matmul_3d_2d(x_q, w_q)?; // [B, T_q, D_head]
    let k = matmul_3d_2d(x_k, w_k)?; // [B, T_k, D_head]
    let v = matmul_3d_2d(x_v, w_v)?; // [B, T_k, D_head]
    let ov = matmul_3d_2d(&v, w_o)?; // [B, T_k, D_model]

    let scores = (q.matmul(&k.transpose(1, 2)?)? * scale)?;
    let attn_weights = candle_nn::ops::softmax(&scores, 2)?;

    let out = attn_weights.matmul(&ov)?;
    Ok((out, attn_weights))
}

/// Prepends register tokens and computes Multi-Token Prediction (MTP) heads.
pub fn register_prepend_and_mtp(
    x_seq: &Tensor,
    registers: &Tensor,
    w_mtp: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let x_aug = Tensor::cat(&[registers, x_seq], 1)?;
    let seq_len = x_seq.dims()[1];
    let x_last = x_seq.i((.., seq_len - 1, ..))?; // [B, D_model]

    let k_future = w_mtp.dims()[0];
    let mut outs = Vec::with_capacity(k_future);

    for k in 0..k_future {
        let wk = w_mtp.i(k)?; // [D_vocab, D_model]
        let logits_k = x_last.matmul(&wk.t()?)?; // [B, D_vocab]
        outs.push(logits_k.unsqueeze(1)?);
    }

    let mtp_logits = Tensor::cat(&outs, 1)?;
    Ok((x_aug, mtp_logits))
}
