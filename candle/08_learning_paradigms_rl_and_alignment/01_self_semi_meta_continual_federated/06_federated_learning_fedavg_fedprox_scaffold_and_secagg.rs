//! Federated Learning Fedavg Fedprox Scaffold and Secagg
//! in candle/08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// FedAvg weighted parameter aggregation: sum_k (n_k / N) * w_k.
pub fn fedavg_aggregate(client_weights: &Tensor, client_counts: &Tensor) -> Result<Tensor> {
    let cw = client_weights.to_vec2::<f32>()?;
    let cc = client_counts.to_vec1::<f32>()?;
    let k = cw.len();
    let d = cw[0].len();

    let total_n = cc.iter().sum::<f32>();
    let mut out = vec![0.0f32; d];

    for i in 0..k {
        let weight = cc[i] / total_n;
        for j in 0..d {
            out[j] += weight * cw[i][j];
        }
    }

    Tensor::from_vec(out, (d,), client_weights.device())
}

/// FedProx local update with proximal constraint: w_local - eta * (grad + mu * (w_local - w_global)).
pub fn fedprox_gradient_step(
    w_local: &Tensor,
    w_global: &Tensor,
    grad_local: &Tensor,
    eta: f64,
    mu: f64,
) -> Result<Tensor> {
    let diff = (w_local - w_global)?;
    let prox_grad = (grad_local + (diff * mu)?)?;
    w_local - (prox_grad * eta)?
}

/// SCAFFOLD client drift correction: grad_i - c_i + c_global.
pub fn scaffold_correct_gradient(
    grad_i: &Tensor,
    c_i: &Tensor,
    c_global: &Tensor,
) -> Result<Tensor> {
    (grad_i - c_i)?.broadcast_add(c_global)
}

/// Secure Aggregation: summing across all clients cancels out pairwise additive masks.
pub fn secagg_aggregate_sum(masked_models: &Tensor) -> Result<Tensor> {
    masked_models.sum(0)
}
