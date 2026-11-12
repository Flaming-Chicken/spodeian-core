//! GRPO Dapo and Verifiable Reward Reasoning
//! in candle/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Composite Verifiable Rule-Based Rewards (Accuracy + Formatting).
pub fn composite_verifiable_reward(
    accuracy_flags: &Tensor,
    format_flags: &Tensor,
    acc_weight: f32,
    format_weight: f32,
) -> Result<Tensor> {
    let acc = (accuracy_flags * (acc_weight as f64))?;
    let fmt = (format_flags * (format_weight as f64))?;
    acc + fmt
}

/// GRPO Group-Normalized Advantage Estimation.
pub fn grpo_group_advantages(rewards: &Tensor, eps: f32) -> Result<Tensor> {
    let rew = rewards.to_vec2::<f32>()?;
    let b = rew.len();
    let g = rew[0].len();
    let mut flat_adv = Vec::with_capacity(b * g);

    for bi in 0..b {
        let mut sum_r = 0.0f32;
        for gi in 0..g {
            sum_r += rew[bi][gi];
        }
        let mean = sum_r / (g as f32);

        let mut var_sum = 0.0f32;
        for gi in 0..g {
            let diff = rew[bi][gi] - mean;
            var_sum += diff * diff;
        }
        let std = (var_sum / (g as f32) + eps).sqrt();

        for gi in 0..g {
            flat_adv.push((rew[bi][gi] - mean) / std);
        }
    }

    Tensor::from_vec(flat_adv, (b, g), rewards.device())
}

/// Schulman's k3 unbiased low-variance non-negative KL divergence.
pub fn k3_kl_divergence(log_probs_policy: &Tensor, log_probs_ref: &Tensor) -> Result<Tensor> {
    let p = log_probs_policy.to_vec3::<f32>()?;
    let r = log_probs_ref.to_vec3::<f32>()?;

    let b = p.len();
    let g = p[0].len();
    let t = p[0][0].len();
    let mut flat_kl = Vec::with_capacity(b * g * t);

    for bi in 0..b {
        for gi in 0..g {
            for ti in 0..t {
                let delta = p[bi][gi][ti] - r[bi][gi][ti];
                let kl = (-delta).exp() - 1.0f32 + delta;
                flat_kl.push(kl);
            }
        }
    }

    Tensor::from_vec(flat_kl, (b, g, t), log_probs_policy.device())
}

/// GRPO Clipped Objective with reference KL penalty.
pub fn grpo_clipped_loss(
    log_probs_policy: &Tensor,
    log_probs_old: &Tensor,
    advantages: &Tensor,
    kl_div: &Tensor,
    clip_eps: f32,
    beta: f32,
) -> Result<Tensor> {
    let p = log_probs_policy.to_vec3::<f32>()?;
    let old = log_probs_old.to_vec3::<f32>()?;
    let adv = advantages.to_vec2::<f32>()?;
    let kl = kl_div.to_vec3::<f32>()?;

    let b = p.len();
    let g = p[0].len();
    let t = p[0][0].len();
    let mut total = 0.0f32;

    for bi in 0..b {
        for gi in 0..g {
            let a = adv[bi][gi];
            for ti in 0..t {
                let ratio = (p[bi][gi][ti] - old[bi][gi][ti]).exp();
                let surr1 = ratio * a;
                let clamped_ratio = ratio.clamp(1.0f32 - clip_eps, 1.0f32 + clip_eps);
                let surr2 = clamped_ratio * a;
                let ppo_obj = surr1.min(surr2);
                total += ppo_obj - beta * kl[bi][gi][ti];
            }
        }
    }

    Tensor::new(-total / ((b * g * t) as f32), log_probs_policy.device())
}
