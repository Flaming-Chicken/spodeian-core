//! Structured State Space Models S4 and LRU
//! in candle/05_architectures/03_recurrent_ssm_and_memory.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Linear Recurrent Unit (LRU) recurrent forward pass in complex domain.
pub fn lru_recurrent_forward(
    x_seq: &Tensor,
    nu: &Tensor,
    theta: &Tensor,
    b_re: &Tensor,
    b_im: &Tensor,
    c_re: &Tensor,
    c_im: &Tensor,
    d_skip: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (b, t, _) = x_seq.dims3()?;
    let d_state = nu.dim(0)?;
    let dev = x_seq.device();

    // diag_mag = exp(-clamp(nu, 1e-4, f32::MAX))
    let nu_clamped = nu.clamp(1e-4f32, f32::MAX)?;
    let diag_mag = nu_clamped.neg()?.exp()?;
    let diag_re = diag_mag.mul(&theta.cos()?)?;
    let diag_im = diag_mag.mul(&theta.sin()?)?;

    let mut h_re = Tensor::zeros((b, d_state), candle_core::DType::F32, dev)?;
    let mut h_im = Tensor::zeros((b, d_state), candle_core::DType::F32, dev)?;

    let mut outs = Vec::with_capacity(t);

    let b_re_t = b_re.t()?;
    let b_im_t = b_im.t()?;
    let c_re_t = c_re.t()?;
    let c_im_t = c_im.t()?;

    for ti in 0..t {
        let xt = x_seq.narrow(1, ti, 1)?.squeeze(1)?;

        let bu_re = xt.matmul(&b_re_t)?;
        let bu_im = xt.matmul(&b_im_t)?;

        // h_re_next = diag_re * h_re - diag_im * h_im + bu_re
        let term_re1 = h_re.broadcast_mul(&diag_re)?;
        let term_re2 = h_im.broadcast_mul(&diag_im)?;
        let h_re_next = term_re1.sub(&term_re2)?.add(&bu_re)?;

        // h_im_next = diag_re * h_im + diag_im * h_re + bu_im
        let term_im1 = h_im.broadcast_mul(&diag_re)?;
        let term_im2 = h_re.broadcast_mul(&diag_im)?;
        let h_im_next = term_im1.add(&term_im2)?.add(&bu_im)?;

        h_re = h_re_next;
        h_im = h_im_next;

        // y_t = h_re @ c_re.t() - h_im @ c_im.t() + xt * d_skip
        let y_re1 = h_re.matmul(&c_re_t)?;
        let y_re2 = h_im.matmul(&c_im_t)?;
        let y_skip = xt.broadcast_mul(d_skip)?;
        let y_t = y_re1.sub(&y_re2)?.add(&y_skip)?;

        outs.push(y_t.unsqueeze(1)?);
    }

    let t_refs: Vec<&Tensor> = outs.iter().collect();
    let seq_out = Tensor::cat(&t_refs, 1)?;
    Ok((seq_out, h_re, h_im))
}
