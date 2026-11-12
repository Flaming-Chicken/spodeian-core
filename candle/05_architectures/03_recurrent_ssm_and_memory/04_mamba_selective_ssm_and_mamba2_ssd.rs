//! Mamba Selective SSM and Mamba2 SSD
//! in candle/05_architectures/03_recurrent_ssm_and_memory.

use candle_core::{IndexOp, Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Numerically stable softplus: ln(1 + exp(x)).
pub fn softplus(x: &Tensor) -> Result<Tensor> {
    let dims = x.shape();
    let flat = x.flatten_all()?.to_vec1::<f32>()?;
    let mut out = Vec::with_capacity(flat.len());
    for v in flat {
        if v > 20.0 {
            out.push(v);
        } else if v < -20.0 {
            out.push(v.exp());
        } else {
            out.push((1.0f32 + v.exp()).ln());
        }
    }
    Tensor::from_vec(out, dims, x.device())
}

/// Selective State Space Model (Mamba-1) recurrent unrolling.
pub fn mamba_selective_forward(
    x_seq: &Tensor,
    a_log: &Tensor,
    w_b: &Tensor,
    w_c: &Tensor,
    w_delta: &Tensor,
    b_delta: &Tensor,
    d_skip: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let dims = x_seq.dims();
    let b_sz = dims[0];
    let seq_len = dims[1];
    let d_model = dims[2];
    let d_state = a_log.dims()[1];

    let a_real = a_log.exp()?.neg()?; // [D_model, D_state]
    let mut h_t = Tensor::zeros(
        (b_sz, d_model, d_state),
        candle_core::DType::F32,
        x_seq.device(),
    )?;
    let mut outs = Vec::with_capacity(seq_len);

    for t in 0..seq_len {
        let xt = x_seq.i((.., t, ..))?; // [B, D_model]
        let bt = xt.matmul(&w_b.t()?)?; // [B, D_state]
        let ct = xt.matmul(&w_c.t()?)?; // [B, D_state]

        let delta_pre = xt.matmul(&w_delta.t()?)?.broadcast_add(b_delta)?;
        let delta_t = softplus(&delta_pre)?; // [B, D_model]

        // a_bar = exp(delta_t.unsqueeze(-1) * a_real.unsqueeze(0))
        let dt_a = delta_t.unsqueeze(2)?.broadcast_mul(&a_real.unsqueeze(0)?)?;
        let a_bar = dt_a.exp()?; // [B, D_model, D_state]

        // b_bar = delta_t.unsqueeze(-1) * bt.unsqueeze(1)
        let b_bar = delta_t.unsqueeze(2)?.broadcast_mul(&bt.unsqueeze(1)?)?; // [B, D_model, D_state]

        // h_t = a_bar * h_t + b_bar * xt.unsqueeze(-1)
        let h_next = (a_bar.broadcast_mul(&h_t)? + b_bar.broadcast_mul(&xt.unsqueeze(2)?)?)?;
        h_t = h_next;

        // y_t = sum(h_t * ct.unsqueeze(1), dim=-1) + xt * d_skip
        let y_t = (h_t.broadcast_mul(&ct.unsqueeze(1)?)?.sum(2)? + xt.broadcast_mul(d_skip)?)?;
        outs.push(y_t.unsqueeze(1)?);
    }

    let seq_out = Tensor::cat(&outs, 1)?;
    Ok((seq_out, h_t))
}

/// Computes the 1-semiseparable causal decay matrix L for Mamba-2 SSD.
pub fn mamba2_ssd_mask(delta_seq: &Tensor, a_scalar: &Tensor) -> Result<Tensor> {
    let dims = delta_seq.dims();
    let seq_len = dims[1];

    let softplus_delta = softplus(delta_seq)?;
    let a_scalar_abs = a_scalar.abs()?;
    let decay = softplus_delta.broadcast_mul(&a_scalar_abs)?.neg()?;

    // Cumsum along sequence dimension via upper-triangular matrix multiplication:
    // [x0, x1, x2] @ [[1, 1, 1], [0, 1, 1], [0, 0, 1]] = [x0, x0+x1, x0+x1+x2]
    let mut triu_vals = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in i..seq_len {
            triu_vals[i * seq_len + j] = 1.0;
        }
    }
    let triu_tensor = Tensor::from_vec(triu_vals, (seq_len, seq_len), delta_seq.device())?;
    let cumsum_decay = decay.matmul(&triu_tensor)?; // [B, T]

    // diff[b, i, j] = cumsum[b, i] - cumsum[b, j]
    let diff = cumsum_decay
        .unsqueeze(2)?
        .broadcast_sub(&cumsum_decay.unsqueeze(1)?)?; // [B, T, T]
    let exp_diff = diff.exp()?;

    // Lower triangular mask (causal: i >= j)
    let mut mask_vals = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..=i {
            mask_vals[i * seq_len + j] = 1.0;
        }
    }
    let mask_t = Tensor::from_vec(mask_vals, (seq_len, seq_len), delta_seq.device())?;
    let l_matrix = exp_diff.broadcast_mul(&mask_t.unsqueeze(0)?)?;

    Ok(l_matrix)
}
