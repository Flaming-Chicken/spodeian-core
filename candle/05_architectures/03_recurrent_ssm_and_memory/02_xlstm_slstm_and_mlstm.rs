//! xLSTM sLSTM and mLSTM
//! in candle/05_architectures/03_recurrent_ssm_and_memory.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// sLSTM recurrent forward pass.
pub fn slstm_forward(
    x_seq: &Tensor,
    w_i: &Tensor,
    w_f: &Tensor,
    w_z: &Tensor,
    w_o: &Tensor,
    u_i: &Tensor,
    u_f: &Tensor,
    u_z: &Tensor,
    u_o: &Tensor,
    b_i: &Tensor,
    b_f: &Tensor,
    b_z: &Tensor,
    b_o: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (b, t, _) = x_seq.dims3()?;
    let d_hid = w_i.dim(0)?;
    let dev = x_seq.device();

    let mut c_s = Tensor::zeros((b, d_hid), candle_core::DType::F32, dev)?;
    let mut n_s = Tensor::zeros((b, d_hid), candle_core::DType::F32, dev)?;
    let mut m_s = Tensor::zeros((b, d_hid), candle_core::DType::F32, dev)?;
    let mut h_s = Tensor::zeros((b, d_hid), candle_core::DType::F32, dev)?;

    let mut outs = Vec::with_capacity(t);

    let w_i_t = w_i.t()?;
    let w_f_t = w_f.t()?;
    let w_z_t = w_z.t()?;
    let w_o_t = w_o.t()?;
    let u_i_t = u_i.t()?;
    let u_f_t = u_f.t()?;
    let u_z_t = u_z.t()?;
    let u_o_t = u_o.t()?;

    for ti in 0..t {
        let xt = x_seq.narrow(1, ti, 1)?.squeeze(1)?;

        let raw_i = xt
            .matmul(&w_i_t)?
            .add(&h_s.matmul(&u_i_t)?)?
            .broadcast_add(b_i)?;
        let raw_f = xt
            .matmul(&w_f_t)?
            .add(&h_s.matmul(&u_f_t)?)?
            .broadcast_add(b_f)?;
        let raw_z = xt
            .matmul(&w_z_t)?
            .add(&h_s.matmul(&u_z_t)?)?
            .broadcast_add(b_z)?;
        let raw_o = xt
            .matmul(&w_o_t)?
            .add(&h_s.matmul(&u_o_t)?)?
            .broadcast_add(b_o)?;

        let sig_f = candle_nn::ops::sigmoid(&raw_f)?;
        let eps = Tensor::new(1e-12f32, dev)?.broadcast_as(sig_f.shape())?;
        let log_sig_f = sig_f.add(&eps)?.log()?;
        let m_next = log_sig_f.add(&m_s)?.maximum(&raw_i)?;

        let i_gate = raw_i.sub(&m_next)?.exp()?;
        let f_gate = sig_f.mul(&m_s.sub(&m_next)?.exp()?)?;
        let z_cand = raw_z.tanh()?;
        let o_gate = candle_nn::ops::sigmoid(&raw_o)?;

        c_s = f_gate.mul(&c_s)?.add(&i_gate.mul(&z_cand)?)?;
        n_s = f_gate.mul(&n_s)?.add(&i_gate)?;
        m_s = m_next;

        let n_eps = Tensor::new(1e-6f32, dev)?.broadcast_as(n_s.shape())?;
        let h_tilde = c_s.div(&n_s.add(&n_eps)?)?;
        h_s = o_gate.mul(&h_tilde)?;
        outs.push(h_s.unsqueeze(1)?);
    }

    let t_refs: Vec<&Tensor> = outs.iter().collect();
    let seq_out = Tensor::cat(&t_refs, 1)?;
    Ok((seq_out, c_s, n_s))
}

/// mLSTM recurrent forward pass.
pub fn mlstm_forward(
    x_seq: &Tensor,
    w_q: &Tensor,
    w_k: &Tensor,
    w_v: &Tensor,
    w_i: &Tensor,
    w_f: &Tensor,
    w_o: &Tensor,
    b_q: &Tensor,
    b_k: &Tensor,
    b_v: &Tensor,
    b_i: &Tensor,
    b_f: &Tensor,
    b_o: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (b, t, _) = x_seq.dims3()?;
    let d_hid = w_q.dim(0)?;
    let dev = x_seq.device();

    let mut c_m = Tensor::zeros((b, d_hid, d_hid), candle_core::DType::F32, dev)?;
    let mut n_m = Tensor::zeros((b, d_hid), candle_core::DType::F32, dev)?;
    let mut m_m = Tensor::zeros((b, 1), candle_core::DType::F32, dev)?;

    let mut outs = Vec::with_capacity(t);

    let w_q_t = w_q.t()?;
    let w_k_t = w_k.t()?;
    let w_v_t = w_v.t()?;
    let w_i_t = w_i.t()?;
    let w_f_t = w_f.t()?;
    let w_o_t = w_o.t()?;

    for ti in 0..t {
        let xt = x_seq.narrow(1, ti, 1)?.squeeze(1)?;
        let q_t = xt.matmul(&w_q_t)?.broadcast_add(b_q)?;
        let k_t = xt.matmul(&w_k_t)?.broadcast_add(b_k)?;
        let v_t = xt.matmul(&w_v_t)?.broadcast_add(b_v)?;

        let raw_i = xt.matmul(&w_i_t)?.broadcast_add(b_i)?.mean_keepdim(1)?;
        let raw_f = xt.matmul(&w_f_t)?.broadcast_add(b_f)?.mean_keepdim(1)?;
        let raw_o = xt.matmul(&w_o_t)?.broadcast_add(b_o)?;

        let sig_f = candle_nn::ops::sigmoid(&raw_f)?;
        let eps = Tensor::new(1e-12f32, dev)?.broadcast_as(sig_f.shape())?;
        let log_sig_f = sig_f.add(&eps)?.log()?;
        let m_next = log_sig_f.add(&m_m)?.maximum(&raw_i)?;

        let i_gate = raw_i.sub(&m_next)?.exp()?;
        let f_gate = sig_f.mul(&m_m.sub(&m_next)?.exp()?)?;
        let o_gate = candle_nn::ops::sigmoid(&raw_o)?;

        // vk = v_t.unsqueeze(-1) @ k_t.unsqueeze(1) -> [B, D, D]
        let vk = v_t.unsqueeze(2)?.matmul(&k_t.unsqueeze(1)?)?;

        let f_gate_3d = f_gate.unsqueeze(2)?;
        let i_gate_3d = i_gate.unsqueeze(2)?;
        c_m = f_gate_3d
            .broadcast_mul(&c_m)?
            .add(&i_gate_3d.broadcast_mul(&vk)?)?;
        n_m = f_gate
            .broadcast_mul(&n_m)?
            .add(&i_gate.broadcast_mul(&k_t)?)?;
        m_m = m_next;

        // C_q = C_m @ q_t.unsqueeze(-1) -> [B, D]
        let c_q = c_m.matmul(&q_t.unsqueeze(2)?)?.squeeze(2)?;

        // denom = abs(n_m.unsqueeze(1) @ q_t.unsqueeze(-1))
        let denom_raw = n_m.unsqueeze(1)?.matmul(&q_t.unsqueeze(2)?)?.squeeze(2)?;
        let denom = denom_raw.abs()?.clamp(1.0f32, f32::MAX)?;

        let h_tilde = c_q.broadcast_div(&denom)?;
        let h_m = o_gate.mul(&h_tilde)?;
        outs.push(h_m.unsqueeze(1)?);
    }

    let t_refs: Vec<&Tensor> = outs.iter().collect();
    let seq_out = Tensor::cat(&t_refs, 1)?;
    Ok((seq_out, c_m, n_m))
}
