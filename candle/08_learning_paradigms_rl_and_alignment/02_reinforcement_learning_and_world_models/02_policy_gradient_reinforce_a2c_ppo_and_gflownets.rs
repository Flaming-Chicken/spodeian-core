//! Policy Gradient Reinforce A2c PPO and GFlowNets
//! in candle/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// PPO-Clip policy surrogate loss.
pub fn ppo_clip_loss(
    log_probs: &Tensor,
    old_log_probs: &Tensor,
    advantages: &Tensor,
    clip_eps: f32,
) -> Result<Tensor> {
    let lp = log_probs.to_vec1::<f32>()?;
    let old_lp = old_log_probs.to_vec1::<f32>()?;
    let adv = advantages.to_vec1::<f32>()?;
    let b = lp.len();
    let mut sum_loss = 0.0f32;

    for i in 0..b {
        let ratio = (lp[i] - old_lp[i]).exp();
        let surr1 = ratio * adv[i];
        let clamped_r = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps);
        let surr2 = clamped_r * adv[i];
        let clipped = surr1.min(surr2);
        sum_loss += -clipped;
    }

    Tensor::new(sum_loss / (b as f32), log_probs.device())
}

/// Generalized Advantage Estimation (GAE).
pub fn compute_gae(
    rewards: &Tensor,
    values: &Tensor,
    dones: &Tensor,
    next_value: f32,
    gamma: f32,
    lambda_: f32,
) -> Result<Tensor> {
    let r = rewards.to_vec1::<f32>()?;
    let v = values.to_vec1::<f32>()?;
    let d = dones.to_vec1::<f32>()?;
    let t_len = r.len();
    let mut adv = vec![0.0f32; t_len];
    let mut last_gae = 0.0f32;

    for t in (0..t_len).rev() {
        let v_next = if t == t_len - 1 { next_value } else { v[t + 1] };
        let not_done = 1.0 - d[t];
        let delta = r[t] + gamma * not_done * v_next - v[t];
        last_gae = delta + gamma * lambda_ * not_done * last_gae;
        adv[t] = last_gae;
    }

    Tensor::from_vec(adv, (t_len,), rewards.device())
}

/// GFlowNet Trajectory Balance (TB) objective.
pub fn gflownet_trajectory_balance_loss(
    log_z: &Tensor,
    log_pf: &Tensor,
    log_pb: &Tensor,
    reward: &Tensor,
) -> Result<Tensor> {
    let lz = log_z.to_scalar::<f32>()?;
    let pf = log_pf.to_vec2::<f32>()?;
    let pb = log_pb.to_vec2::<f32>()?;
    let rew = reward.to_vec1::<f32>()?;
    let b = pf.len();
    let mut total_tb_sq = 0.0f32;

    for i in 0..b {
        let sum_pf: f32 = pf[i].iter().sum();
        let sum_pb: f32 = pb[i].iter().sum();
        let log_r = rew[i].max(1e-8).ln();
        let err = lz + sum_pf - log_r - sum_pb;
        total_tb_sq += err * err;
    }

    Tensor::new(total_tb_sq / (b as f32), log_pf.device())
}
