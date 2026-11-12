//! Continual Learning EWC Synaptic Intelligence Lwf and Replay
//! in candle/08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Elastic Weight Consolidation quadratic penalty: 0.5 * lambda * sum(F_i * (theta_i - theta_star_i)^2).
pub fn ewc_penalty(
    theta: &Tensor,
    theta_star: &Tensor,
    fisher_diag: &Tensor,
    lambda_ewc: f64,
) -> Result<Tensor> {
    let diff = (theta - theta_star)?;
    let term = (diff.sqr()?.broadcast_mul(fisher_diag)?.sum_all()? * (0.5 * lambda_ewc))?;
    term.reshape(&[1])
}

/// Synaptic Intelligence importance parameter: max(0, omega_k) / (Delta theta_k^2 + xi).
pub fn synaptic_intelligence_omega(
    delta_theta: &Tensor,
    path_integral: &Tensor,
    xi: f32,
) -> Result<Tensor> {
    let pi_vec = path_integral.to_vec1::<f32>()?;
    let dt_vec = delta_theta.to_vec1::<f32>()?;
    let d = pi_vec.len();

    let mut out = Vec::with_capacity(d);
    for i in 0..d {
        let nom = pi_vec[i].max(0.0f32);
        let den = dt_vec[i] * dt_vec[i] + xi;
        out.push(nom / den);
    }

    Tensor::from_vec(out, (d,), path_integral.device())
}

/// Learning without Forgetting (LwF) knowledge distillation loss under temperature scaling.
pub fn lwf_distillation_loss(
    new_logits: &Tensor,
    old_logits: &Tensor,
    temperature: f64,
) -> Result<Tensor> {
    let b = new_logits.dim(0)?;
    let t_inv = 1.0 / temperature;

    let p_old = candle_nn::ops::softmax(&((old_logits * t_inv)?), 1)?;
    let log_p_new = candle_nn::ops::log_softmax(&((new_logits * t_inv)?), 1)?;
    let log_p_old = candle_nn::ops::log_softmax(&((old_logits * t_inv)?), 1)?;

    let kl_sum = (p_old * (log_p_old - log_p_new)?)?.sum_all()?;
    let kl_mean = (kl_sum * (1.0 / (b as f64)))?;
    let loss = (kl_mean * (temperature * temperature))?;
    loss.reshape(&[1])
}

/// Dark Experience Replay (DER++) loss: alpha * MSE(current, past) + beta * CE(current, labels).
pub fn der_dark_replay_loss(
    current_logits: &Tensor,
    past_logits: &Tensor,
    labels: &Tensor,
    alpha: f64,
    beta: f64,
) -> Result<Tensor> {
    let mse = ((current_logits - past_logits)?.sqr()?.mean_all()? * alpha)?;
    let log_p = candle_nn::ops::log_softmax(current_logits, 1)?;
    let log_p_vec = log_p.to_vec2::<f32>()?;
    let labels_vec = labels.to_vec1::<i64>()?;

    let b = labels_vec.len();
    let mut ce_sum = 0.0f32;
    for i in 0..b {
        let label_idx = labels_vec[i] as usize;
        ce_sum += -log_p_vec[i][label_idx];
    }
    let ce = (ce_sum / (b as f32)) as f64;

    let total = mse.to_scalar::<f32>()? as f64 + (beta * ce);
    Tensor::new(&[total as f32], current_logits.device())
}
