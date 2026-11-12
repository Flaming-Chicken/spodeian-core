//! Reservoir Computing, Echo State Networks, LSM, and NG-RC
//! in candle/05_architectures/03_recurrent_ssm_and_memory.

use candle_core::{IndexOp, Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Leaky Echo State Network recurrent forward pass.
pub fn esn_forward(
    x_seq: &Tensor,
    w_in: &Tensor,
    w_res: &Tensor,
    b_res: &Tensor,
    alpha: f64,
    w_out: &Tensor,
    b_out: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let dims = x_seq.dims();
    let b_sz = dims[0];
    let seq_len = dims[1];
    let d_res = w_in.dims()[0];

    let mut h_t = Tensor::zeros((b_sz, d_res), candle_core::DType::F32, x_seq.device())?;
    let mut outs = Vec::with_capacity(seq_len);

    for t in 0..seq_len {
        let xt = x_seq.i((.., t, ..))?; // [B, D_in]
        let in_proj = xt.matmul(&w_in.t()?)?;
        let res_proj = h_t.matmul(&w_res.t()?)?;
        let pre_act = in_proj.broadcast_add(&res_proj)?.broadcast_add(b_res)?;
        let tilde_h = pre_act.tanh()?;

        let h_decayed = (h_t * (1.0 - alpha))?;
        let h_injected = (tilde_h * alpha)?;
        h_t = (h_decayed + h_injected)?;

        let yt = h_t.matmul(&w_out.t()?)?.broadcast_add(b_out)?;
        outs.push(yt.unsqueeze(1)?);
    }

    let seq_out = Tensor::cat(&outs, 1)?;
    Ok((seq_out, h_t))
}

/// Next-Generation Reservoir Computing (NG-RC) nonlinear feature expansion.
pub fn ngrc_features(x_curr: &Tensor, x_prev: &Tensor) -> Result<Tensor> {
    let lin = Tensor::cat(&[x_curr, x_prev], 1)?; // [B, 2 * D]
    let quad = x_curr
        .unsqueeze(2)?
        .matmul(&x_prev.unsqueeze(1)?)?
        .flatten_from(1)?; // [B, D * D]
    Tensor::cat(&[&lin, &quad], 1)
}
