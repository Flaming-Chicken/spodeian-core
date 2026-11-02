//! Linear Attention, RWKV, RetNet, GLA, DeltaNet, and TTT
//! in candle/05_architectures/03_recurrent_ssm_and_memory.

use candle_core::{IndexOp, Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Recurrent Retention (RetNet) inference mode.
pub fn retnet_recurrent_forward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    gamma: f64,
) -> Result<(Tensor, Tensor)> {
    let dims = q.dims();
    let b_sz = dims[0];
    let seq_len = dims[1];
    let d_k = dims[2];
    let d_v = v.dims()[2];

    let mut s_t = Tensor::zeros((b_sz, d_k, d_v), candle_core::DType::F32, q.device())?;
    let mut outs = Vec::with_capacity(seq_len);

    for t in 0..seq_len {
        let qt = q.i((.., t, ..))?.unsqueeze(1)?; // [B, 1, D_k]
        let kt = k.i((.., t, ..))?.unsqueeze(2)?; // [B, D_k, 1]
        let vt = v.i((.., t, ..))?.unsqueeze(1)?; // [B, 1, D_v]

        let kv = kt.matmul(&vt)?; // [B, D_k, D_v]
        let s_decayed = (s_t * gamma)?;
        s_t = (s_decayed + kv)?;

        let ot = qt.matmul(&s_t)?.squeeze(1)?; // [B, D_v]
        outs.push(ot.unsqueeze(1)?);
    }

    let seq_out = Tensor::cat(&outs, 1)?;
    Ok((seq_out, s_t))
}

/// Parallel Retention (RetNet) training mode.
pub fn retnet_parallel_forward(q: &Tensor, k: &Tensor, v: &Tensor, gamma: f64) -> Result<Tensor> {
    let dims = q.dims();
    let seq_len = dims[1];

    let scores = q.matmul(&k.transpose(1, 2)?)?; // [B, T, T]

    let mut decay_vals = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..=i {
            let diff = (i - j) as i32;
            decay_vals[i * seq_len + j] = (gamma as f32).powi(diff);
        }
    }
    let decay_tensor = Tensor::from_vec(decay_vals, (seq_len, seq_len), q.device())?;
    let retention_matrix = scores.broadcast_mul(&decay_tensor.unsqueeze(0)?)?;
    retention_matrix.matmul(v)
}

/// DeltaNet online associative memory update.
pub fn deltanet_forward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    beta: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let dims = q.dims();
    let b_sz = dims[0];
    let seq_len = dims[1];
    let d_k = dims[2];
    let d_v = v.dims()[2];

    let mut s_t = Tensor::zeros((b_sz, d_k, d_v), candle_core::DType::F32, q.device())?;
    let mut outs = Vec::with_capacity(seq_len);

    for t in 0..seq_len {
        let qt = q.i((.., t, ..))?.unsqueeze(1)?; // [B, 1, D_k]
        let kt = k.i((.., t, ..))?.unsqueeze(2)?; // [B, D_k, 1]
        let kt_t = kt.transpose(1, 2)?; // [B, 1, D_k]
        let vt = v.i((.., t, ..))?.unsqueeze(1)?; // [B, 1, D_v]
        let b_val = beta.i((.., t))?.unsqueeze(1)?.unsqueeze(2)?; // [B, 1, 1]

        let pred_v = kt_t.matmul(&s_t)?; // [B, 1, D_v]
        let err_v = vt.sub(&pred_v)?; // [B, 1, D_v]
        let delta = kt.matmul(&err_v)?; // [B, D_k, D_v]
        let delta_scaled = delta.broadcast_mul(&b_val)?;

        s_t = (s_t + delta_scaled)?;
        let ot = qt.matmul(&s_t)?.squeeze(1)?; // [B, D_v]
        outs.push(ot.unsqueeze(1)?);
    }

    let seq_out = Tensor::cat(&outs, 1)?;
    Ok((seq_out, s_t))
}
