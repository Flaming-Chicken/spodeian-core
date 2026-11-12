//! RLHF Reward Modeling and PPO vs Rloo
//! in candle/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Bradley-Terry Preference Reward Loss: -ln sigma(r_w - r_l) = softplus(-(r_w - r_l)).
pub fn bradley_terry_reward_loss(
    rewards_chosen: &Tensor,
    rewards_rejected: &Tensor,
) -> Result<Tensor> {
    let rw = rewards_chosen.to_vec1::<f32>()?;
    let rl = rewards_rejected.to_vec1::<f32>()?;
    let b = rw.len();
    let mut total = 0.0f32;

    for i in 0..b {
        let diff = rw[i] - rl[i];
        // softplus(-diff) = ln(1 + exp(-diff))
        let sp = if -diff > 30.0 {
            -diff
        } else {
            (1.0f32 + (-diff).exp()).ln()
        };
        total += sp;
    }

    Tensor::new(total / (b as f32), rewards_chosen.device())
}

/// RLHF Token-Level Reward shaping with reference policy KL penalty.
pub fn rlhf_token_level_rewards(
    log_probs_policy: &Tensor,
    log_probs_ref: &Tensor,
    terminal_scores: &Tensor,
    kl_coeff: f32,
) -> Result<Tensor> {
    let p_lp = log_probs_policy.to_vec2::<f32>()?;
    let r_lp = log_probs_ref.to_vec2::<f32>()?;
    let term = terminal_scores.to_vec1::<f32>()?;

    let b = p_lp.len();
    let t = p_lp[0].len();
    let mut flat_rew = Vec::with_capacity(b * t);

    for bi in 0..b {
        for ti in 0..t {
            let token_kl = p_lp[bi][ti] - r_lp[bi][ti];
            let mut rew = -kl_coeff * token_kl;
            if ti == t - 1 {
                rew += term[bi];
            }
            flat_rew.push(rew);
        }
    }

    Tensor::from_vec(flat_rew, (b, t), log_probs_policy.device())
}

/// RLOO Leave-One-Out advantage calculation.
pub fn rloo_leave_one_out_advantages(scores: &Tensor) -> Result<Tensor> {
    let s = scores.to_vec2::<f32>()?;
    let b = s.len();
    let k = s[0].len();
    let mut flat_adv = Vec::with_capacity(b * k);

    for bi in 0..b {
        let mut total = 0.0f32;
        for ki in 0..k {
            total += s[bi][ki];
        }

        let denom = (k - 1) as f32;
        for ki in 0..k {
            let baseline = (total - s[bi][ki]) / denom;
            flat_adv.push(s[bi][ki] - baseline);
        }
    }

    Tensor::from_vec(flat_adv, (b, k), scores.device())
}

/// RLOO Policy Gradient Loss: -mean(advantages * log_probs_sum).
pub fn rloo_policy_gradient_loss(advantages: &Tensor, log_probs_sum: &Tensor) -> Result<Tensor> {
    let adv = advantages.to_vec2::<f32>()?;
    let lps = log_probs_sum.to_vec2::<f32>()?;

    let b = adv.len();
    let k = adv[0].len();
    let mut total = 0.0f32;

    for bi in 0..b {
        for ki in 0..k {
            total += -adv[bi][ki] * lps[bi][ki];
        }
    }

    Tensor::new(total / ((b * k) as f32), advantages.device())
}
