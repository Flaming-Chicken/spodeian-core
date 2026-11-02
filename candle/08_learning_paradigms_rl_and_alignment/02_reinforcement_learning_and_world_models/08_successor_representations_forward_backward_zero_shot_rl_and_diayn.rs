//! Successor Representations Forward Backward Zero Shot RL and Diayn
//! in candle/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Successor Features Temporal Difference error calculation.
pub fn successor_features_td_error(
    psi_current: &Tensor,
    features: &Tensor,
    psi_next: &Tensor,
    dones: &Tensor,
    gamma: f32,
) -> Result<Tensor> {
    let psi_c = psi_current.to_vec2::<f32>()?;
    let phi = features.to_vec2::<f32>()?;
    let psi_n = psi_next.to_vec2::<f32>()?;
    let don = dones.to_vec1::<f32>()?;

    let b = psi_c.len();
    let d = psi_c[0].len();
    let mut err_flat = Vec::with_capacity(b * d);

    for bi in 0..b {
        let non_terminal = 1.0f32 - don[bi];
        for di in 0..d {
            let target = phi[bi][di] + gamma * non_terminal * psi_n[bi][di];
            err_flat.push(target - psi_c[bi][di]);
        }
    }

    Tensor::from_vec(err_flat, (b, d), psi_current.device())
}

/// Generalized Policy Improvement (GPI) upper envelope over K policies.
pub fn gpi_transfer_value(psi_policies: &Tensor, reward_weights: &Tensor) -> Result<Tensor> {
    let psi = psi_policies.to_vec3::<f32>()?;
    let w = reward_weights.to_vec1::<f32>()?;

    let b = psi.len();
    let k = psi[0].len();
    let d = psi[0][0].len();
    let mut gpi_vec = Vec::with_capacity(b);

    for bi in 0..b {
        let mut max_q = f32::NEG_INFINITY;
        for ki in 0..k {
            let mut q = 0.0f32;
            for di in 0..d {
                q += psi[bi][ki][di] * w[di];
            }
            if q > max_q {
                max_q = q;
            }
        }
        gpi_vec.push(max_q);
    }

    Tensor::from_vec(gpi_vec, (b,), psi_policies.device())
}

/// Forward-Backward (FB) Zero-Shot RL bilinear value evaluation.
pub fn fb_zero_shot_value(forward_map: &Tensor, task_embedding: &Tensor) -> Result<Tensor> {
    let f = forward_map.to_vec2::<f32>()?;
    let z = task_embedding.to_vec2::<f32>()?;

    let b = f.len();
    let d = f[0].len();
    let mut vals = Vec::with_capacity(b);

    for bi in 0..b {
        let mut dot = 0.0f32;
        for di in 0..d {
            dot += f[bi][di] * z[bi][di];
        }
        vals.push(dot);
    }

    Tensor::from_vec(vals, (b,), forward_map.device())
}

/// DIAYN Skill Intrinsic Reward and Discriminator Cross-Entropy Loss.
pub fn diayn_skill_reward_and_loss(
    logits: &Tensor,
    skill_labels: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let logit_vec = logits.to_vec2::<f32>()?;
    let labels = skill_labels.to_vec1::<i64>()?;

    let b = logit_vec.len();
    let k = logit_vec[0].len();
    let log_prior = -(k as f32).ln();

    let mut rewards = Vec::with_capacity(b);
    let mut total_loss = 0.0f32;

    for bi in 0..b {
        let mut max_logit = f32::NEG_INFINITY;
        for ki in 0..k {
            if logit_vec[bi][ki] > max_logit {
                max_logit = logit_vec[bi][ki];
            }
        }

        let mut sum_exp = 0.0f32;
        for ki in 0..k {
            sum_exp += (logit_vec[bi][ki] - max_logit).exp();
        }
        let log_sum_exp = max_logit + sum_exp.ln();

        let target_skill = labels[bi] as usize;
        let skill_log_prob = logit_vec[bi][target_skill] - log_sum_exp;

        rewards.push(skill_log_prob - log_prior);
        total_loss += -skill_log_prob;
    }

    let rew_tensor = Tensor::from_vec(rewards, (b,), logits.device())?;
    let loss_tensor = Tensor::new(total_loss / (b as f32), logits.device())?;
    Ok((rew_tensor, loss_tensor))
}
