//! RNN, LSTM, GRU & Attention Seq2Seq
//! in candle/05_architectures/03_recurrent_ssm_and_memory.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Recurrent LSTM unrolling forward pass.
pub fn lstm_forward(
    x_seq: &Tensor,
    h0: &Tensor,
    c0: &Tensor,
    w_ih: &Tensor,
    w_hh: &Tensor,
    b_ih: &Tensor,
    b_hh: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (_, t, _) = x_seq.dims3()?;
    let d_hid = h0.dim(1)?;
    let mut ht = h0.clone();
    let mut ct = c0.clone();
    let mut outputs = Vec::with_capacity(t);

    let w_ih_t = w_ih.t()?;
    let w_hh_t = w_hh.t()?;

    for ti in 0..t {
        let xt = x_seq.narrow(1, ti, 1)?.squeeze(1)?;
        let gates = (xt
            .matmul(&w_ih_t)?
            .broadcast_add(b_ih)?
            .add(&ht.matmul(&w_hh_t)?)?
            .broadcast_add(b_hh))?;

        let i_t = candle_nn::ops::sigmoid(&gates.narrow(1, 0, d_hid)?)?;
        let f_t = candle_nn::ops::sigmoid(&gates.narrow(1, d_hid, d_hid)?)?;
        let g_t = gates.narrow(1, 2 * d_hid, d_hid)?.tanh()?;
        let o_t = candle_nn::ops::sigmoid(&gates.narrow(1, 3 * d_hid, d_hid)?)?;

        ct = f_t.mul(&ct)?.add(&i_t.mul(&g_t)?)?;
        ht = o_t.mul(&ct.tanh()?)?;

        outputs.push(ht.unsqueeze(1)?);
    }

    let t_refs: Vec<&Tensor> = outputs.iter().collect();
    let seq_out = Tensor::cat(&t_refs, 1)?;
    Ok((seq_out, ht, ct))
}

/// Recurrent GRU unrolling forward pass.
pub fn gru_forward(
    x_seq: &Tensor,
    h0: &Tensor,
    w_ih: &Tensor,
    w_hh: &Tensor,
    b_ih: &Tensor,
    b_hh: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (_, t, _) = x_seq.dims3()?;
    let d_hid = h0.dim(1)?;
    let mut ht = h0.clone();
    let mut outputs = Vec::with_capacity(t);

    let w_ih_t = w_ih.t()?;
    let w_hh_t = w_hh.t()?;

    for ti in 0..t {
        let xt = x_seq.narrow(1, ti, 1)?.squeeze(1)?;
        let gi = xt.matmul(&w_ih_t)?.broadcast_add(b_ih)?;
        let gh = ht.matmul(&w_hh_t)?.broadcast_add(b_hh)?;

        let r_t =
            candle_nn::ops::sigmoid(&(gi.narrow(1, 0, d_hid)?.add(&gh.narrow(1, 0, d_hid)?)?))?;
        let z_t = candle_nn::ops::sigmoid(
            &(gi.narrow(1, d_hid, d_hid)?
                .add(&gh.narrow(1, d_hid, d_hid)?)?),
        )?;

        let cand_in = gi.narrow(1, 2 * d_hid, d_hid)?;
        let cand_h = r_t.mul(&gh.narrow(1, 2 * d_hid, d_hid)?)?;
        let n_t = cand_in.add(&cand_h)?.tanh()?;

        let ones = Tensor::ones_like(&z_t)?;
        let one_minus_z = ones.sub(&z_t)?;
        ht = one_minus_z.mul(&n_t)?.add(&z_t.mul(&ht)?)?;

        outputs.push(ht.unsqueeze(1)?);
    }

    let t_refs: Vec<&Tensor> = outputs.iter().collect();
    let seq_out = Tensor::cat(&t_refs, 1)?;
    Ok((seq_out, ht))
}

/// Luong Multiplicative Attention forward pass.
pub fn luong_attention_forward(
    decoder_query: &Tensor,
    enc_states: &Tensor,
    w_attn: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let s_proj = decoder_query.matmul(w_attn)?; // [B, D_hid]
    let scores = enc_states.matmul(&s_proj.unsqueeze(2)?)?.squeeze(2)?; // [B, T]
    let attn_weights = candle_nn::ops::softmax_last_dim(&scores)?; // [B, T]
    let context = attn_weights.unsqueeze(1)?.matmul(enc_states)?.squeeze(1)?; // [B, D_hid]
    Ok((context, attn_weights))
}
