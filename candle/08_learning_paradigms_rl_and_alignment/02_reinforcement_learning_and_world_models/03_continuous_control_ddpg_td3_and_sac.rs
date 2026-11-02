//! Continuous Control Ddpg Td3 and SAC
//! in candle/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// TD3 Target Action Smoothing.
pub fn td3_target_smoothing(
    target_actions: &Tensor,
    noise: &Tensor,
    noise_clip: f32,
    action_min: f32,
    action_max: f32,
) -> Result<Tensor> {
    let ta = target_actions.to_vec2::<f32>()?;
    let ns = noise.to_vec2::<f32>()?;
    let b = ta.len();
    let d = ta[0].len();
    let mut out = Vec::with_capacity(b * d);

    for i in 0..b {
        for j in 0..d {
            let n_clamped = ns[i][j].clamp(-noise_clip, noise_clip);
            let a = (ta[i][j] + n_clamped).clamp(action_min, action_max);
            out.push(a);
        }
    }

    Tensor::from_vec(out, (b, d), target_actions.device())
}

/// TD3 Clipped Double-Q Bellman Target.
pub fn td3_clipped_critic_target(
    rewards: &Tensor,
    dones: &Tensor,
    q1: &Tensor,
    q2: &Tensor,
    gamma: f32,
) -> Result<Tensor> {
    let r = rewards.to_vec1::<f32>()?;
    let d = dones.to_vec1::<f32>()?;
    let q1_v = q1.to_vec1::<f32>()?;
    let q2_v = q2.to_vec1::<f32>()?;
    let b = r.len();
    let mut targets = Vec::with_capacity(b);

    for i in 0..b {
        let min_q = q1_v[i].min(q2_v[i]);
        targets.push(r[i] + gamma * (1.0f32 - d[i]) * min_q);
    }

    Tensor::from_vec(targets, (b,), rewards.device())
}

/// SAC Squashed Gaussian policy action and log-probability.
pub fn sac_squashed_gaussian_log_prob(
    u: &Tensor,
    mean: &Tensor,
    log_std: &Tensor,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    let u_v = u.to_vec2::<f32>()?;
    let m_v = mean.to_vec2::<f32>()?;
    let ls_v = log_std.to_vec2::<f32>()?;
    let b = u_v.len();
    let d = u_v[0].len();
    let mut actions = Vec::with_capacity(b * d);
    let mut log_probs = Vec::with_capacity(b);
    let log_2pi = (2.0f32 * std::f32::consts::PI).ln();

    for i in 0..b {
        let mut g_lp = 0.0f32;
        let mut jac = 0.0f32;
        for j in 0..d {
            let std_val = ls_v[i][j].exp();
            let diff = u_v[i][j] - m_v[i][j];
            let norm_diff = diff / std_val;
            g_lp += -0.5f32 * (norm_diff * norm_diff + 2.0f32 * ls_v[i][j] + log_2pi);
            let a = u_v[i][j].tanh();
            actions.push(a);
            let one_minus_a2 = (1.0f32 - a * a).max(eps);
            jac += one_minus_a2.ln();
        }
        log_probs.push(g_lp - jac);
    }

    let act_tensor = Tensor::from_vec(actions, (b, d), u.device())?;
    let lp_tensor = Tensor::from_vec(log_probs, (b,), u.device())?;
    Ok((act_tensor, lp_tensor))
}

/// SAC Temperature Auto-Tuning Loss.
pub fn sac_temperature_loss(
    log_alpha: &Tensor,
    log_probs: &Tensor,
    target_entropy: f32,
) -> Result<Tensor> {
    let la = log_alpha.to_scalar::<f32>()?;
    let lp = log_probs.to_vec1::<f32>()?;
    let alpha = la.exp();
    let mut total = 0.0f32;

    for p in lp.iter() {
        total += -(alpha * (p + target_entropy));
    }

    Tensor::new(total / (lp.len() as f32), log_probs.device())
}
