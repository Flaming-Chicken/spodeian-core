//! Value Based RL Dqn Rainbow and Qr Dqn
//! in candle/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Dueling Network Q-value aggregation:
/// Q(s, a) = V(s) + (A(s, a) - (1 / |A|) * sum_{a'} A(s, a')).
pub fn dueling_q_values(value: &Tensor, advantage: &Tensor) -> Result<Tensor> {
    let mean_adv = advantage.mean_keepdim(1)?;
    let adv_centered = advantage.broadcast_sub(&mean_adv)?;
    value.broadcast_add(&adv_centered)
}

/// Double DQN Bellman target calculation:
/// a* = argmax_{a'} Q_online(s', a')
/// y = r + gamma * (1 - done) * Q_target(s', a*)
pub fn double_dqn_targets(
    rewards: &Tensor,
    dones: &Tensor,
    q_next_online: &Tensor,
    q_next_target: &Tensor,
    gamma: f64,
) -> Result<Tensor> {
    let best_actions = q_next_online.argmax_keepdim(1)?;
    let q_target_selected = q_next_target.gather(&best_actions, 1)?.squeeze(1)?;
    let one = Tensor::ones_like(dones)?;
    let not_done = (one - dones)?;
    let discounted = (q_target_selected * gamma)?;
    let future = (&discounted * &not_done)?;
    rewards + future
}

/// Prioritized Experience Replay (PER) weighted loss:
/// loss = (1 / B) * sum_i w_i * (q_eval_i - y_i)^2
pub fn per_loss(q_eval: &Tensor, targets: &Tensor, is_weights: &Tensor) -> Result<Tensor> {
    let td_errors = (q_eval - targets)?;
    let sq_errors = td_errors.sqr()?;
    let weighted = (is_weights * sq_errors)?;
    weighted.mean_all()
}

/// Quantile Regression DQN (QR-DQN) pairwise quantile Huber loss.
pub fn quantile_huber_loss(
    current_quantiles: &Tensor,
    target_quantiles: &Tensor,
    tau: &Tensor,
    kappa: f32,
) -> Result<Tensor> {
    let cur_vec = current_quantiles.to_vec2::<f32>()?;
    let tgt_vec = target_quantiles.to_vec2::<f32>()?;
    let tau_vec = tau.to_vec1::<f32>()?;

    let b = cur_vec.len();
    let n = cur_vec[0].len();
    let mut total_loss = 0.0f32;

    for bi in 0..b {
        for i in 0..n {
            let tau_i = tau_vec[i];
            let c = cur_vec[bi][i];
            for j in 0..n {
                let t = tgt_vec[bi][j];
                let u = t - c;
                let abs_u = u.abs();
                let huber = if abs_u <= kappa {
                    0.5f32 * u * u
                } else {
                    kappa * (abs_u - 0.5f32 * kappa)
                };
                let indicator = if u < 0.0f32 { 1.0f32 } else { 0.0f32 };
                let weight = (tau_i - indicator).abs();
                total_loss += weight * huber / kappa;
            }
        }
    }

    let mean_loss = total_loss / ((b * n * n) as f32);
    Tensor::new(mean_loss, current_quantiles.device())
}
