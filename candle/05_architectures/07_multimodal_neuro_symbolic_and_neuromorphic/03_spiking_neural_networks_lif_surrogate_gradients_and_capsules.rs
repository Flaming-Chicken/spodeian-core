//! Spiking Neural Networks, LIF, Surrogate Gradients, and Capsules
//! in candle/05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic.

use candle_core::{DType, Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Multi-step Leaky Integrate-and-Fire (LIF) simulation with hard reset.
pub fn lif_simulation(currents: &Tensor, beta: f64, v_th: f64) -> Result<(Tensor, Tensor)> {
    let dims = currents.dims();
    let t_steps = dims[0];
    let b = dims[1];
    let d = dims[2];

    let mut v = Tensor::zeros((b, d), DType::F32, currents.device())?;
    let mut s = Tensor::zeros((b, d), DType::F32, currents.device())?;

    let mut all_v = Vec::with_capacity(t_steps);
    let mut all_s = Vec::with_capacity(t_steps);

    for t in 0..t_steps {
        let current_t = currents.get(t)?;
        let one_minus_s = s.affine(-1.0, 1.0)?;
        let v_decayed = ((v * beta)? * one_minus_s)?;
        v = (v_decayed + current_t)?;
        s = v.ge(v_th)?.to_dtype(DType::F32)?;
        all_v.push(v.clone());
        all_s.push(s.clone());
    }

    let stacked_s = Tensor::stack(&all_s, 0)?;
    let stacked_v = Tensor::stack(&all_v, 0)?;
    Ok((stacked_s, stacked_v))
}

/// Vector squashing activation function for capsule vectors: s / ||s|| * (||s||^2 / (1 + ||s||^2)).
pub fn capsule_squash(s: &Tensor) -> Result<Tensor> {
    let norm_sq = s.sqr()?.sum_keepdim(2)?;
    let eps = Tensor::new(1e-8f32, s.device())?;
    let norm = (norm_sq.broadcast_add(&eps))?.sqrt()?;
    let one_plus_norm_sq = norm_sq.affine(1.0, 1.0)?;
    let scale = norm_sq.div(&one_plus_norm_sq)?;
    let unit = s.broadcast_div(&norm)?;
    scale.broadcast_mul(&unit)
}

/// Dynamic routing between capsules via agreement iterations.
pub fn capsule_dynamic_routing(u_hat: &Tensor, num_iters: usize) -> Result<Tensor> {
    let dims = u_hat.dims();
    let b = dims[0];
    let n_in = dims[1];
    let n_out = dims[2];
    let d_out = dims[3];

    let mut b_logits = Tensor::zeros((b, n_in, n_out), DType::F32, u_hat.device())?;
    let mut v = Tensor::zeros((b, n_out, d_out), DType::F32, u_hat.device())?;

    for i in 0..num_iters {
        let c = candle_nn::ops::softmax(&b_logits, 2)?;
        let c_exp = c.unsqueeze(3)?;
        let s = (c_exp.broadcast_mul(u_hat)?).sum(1)?;
        v = capsule_squash(&s)?;
        if i < num_iters - 1 {
            let v_exp = v.unsqueeze(1)?;
            let agreement = (u_hat.broadcast_mul(&v_exp)?).sum(3)?;
            b_logits = (b_logits + agreement)?;
        }
    }
    Ok(v)
}
