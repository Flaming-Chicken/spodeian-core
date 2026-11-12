//! Energy-Based Models, RBM, and Hopfield Networks
//! in candle/05_architectures/06_graphs_geometric_and_inr.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// RBM free energy marginalizing hidden states: F(v) = -v @ a - sum_j softplus(v @ w + b).
pub fn rbm_free_energy(v: &Tensor, w: &Tensor, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let a_col = a.unsqueeze(1)?;
    let v_bias = v.matmul(&a_col)?;
    let h_pre = v.matmul(w)?.broadcast_add(b)?;
    let h_act = (h_pre.exp()?.affine(1.0, 1.0))?.log()?;
    let h_term = h_act.sum_keepdim(1)?;
    let neg_v_bias = v_bias.affine(-1.0, 0.0)?;
    neg_v_bias - h_term
}

/// Single-step overdamped Langevin dynamics: x_next = x - 0.5*eps*grad_E + sqrt(eps)*noise.
pub fn langevin_dynamics_step(
    x: &Tensor,
    grad_energy: &Tensor,
    step_size: f64,
    noise: &Tensor,
) -> Result<Tensor> {
    let grad_term = grad_energy.affine(-0.5 * step_size, 0.0)?;
    let noise_term = noise.affine(step_size.sqrt(), 0.0)?;
    (x + grad_term)? + noise_term
}

/// Continuous modern Hopfield network energy: E(x) = -1/beta * LSE(beta * M @ x) + 0.5 * ||x||^2.
pub fn modern_hopfield_energy(x: &Tensor, memories: &Tensor, beta: f64) -> Result<Tensor> {
    let sim = x.matmul(&memories.t()?)?;
    let scaled_sim = sim.affine(beta, 0.0)?;
    let max_val = scaled_sim.max_keepdim(1)?;
    let exp_shifted = scaled_sim.broadcast_sub(&max_val)?.exp()?;
    let sum_exp = exp_shifted.sum_keepdim(1)?;
    let lse = (sum_exp.log()? + max_val)?.affine(1.0 / beta, 0.0)?;
    let norm_term = x.sqr()?.sum_keepdim(1)?.affine(0.5, 0.0)?;
    norm_term - lse
}

/// Associative memory pattern retrieval via attention mechanism.
pub fn modern_hopfield_retrieve(query: &Tensor, memories: &Tensor, beta: f64) -> Result<Tensor> {
    let scores = query.matmul(&memories.t()?)?;
    let scaled = scores.affine(beta, 0.0)?;
    let weights = candle_nn::ops::softmax_last_dim(&scaled)?;
    weights.matmul(memories)
}
