//! Safe RL Constrained MDP PPO Lagrangian CVaR and CBF
//! in candle/11_ai_safety_privacy_and_verification.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// PPO-Lagrangian primal policy loss for Constrained MDPs (Altman 1999; Ray et al. 2019).
pub fn ppo_lagrangian_loss(
    logp: &Tensor,
    old_logp: &Tensor,
    adv_reward: &Tensor,
    adv_cost: &Tensor,
    lambda_param: f32,
    clip_eps: f32,
) -> Result<Tensor> {
    let lp = logp.to_vec1::<f32>()?;
    let olp = old_logp.to_vec1::<f32>()?;
    let ar = adv_reward.to_vec1::<f32>()?;
    let ac = adv_cost.to_vec1::<f32>()?;

    let b = lp.len();
    let mut total_reward = 0.0f32;
    let mut total_cost = 0.0f32;

    for i in 0..b {
        let ratio = (lp[i] - olp[i]).exp();
        let surr1 = ratio * ar[i];
        let clamped_ratio = ratio.max(1.0 - clip_eps).min(1.0 + clip_eps);
        let surr2 = clamped_ratio * ar[i];
        total_reward += surr1.min(surr2);
        total_cost += ratio * ac[i];
    }

    let l_reward = total_reward / (b as f32);
    let l_cost = total_cost / (b as f32);
    let total_loss = -l_reward + lambda_param * l_cost;

    Tensor::new(total_loss, logp.device())
}

/// Dual ascent step on Lagrangian safety multiplier.
pub fn lagrangian_multiplier_step(
    lambda_val: f32,
    cost_mean: f32,
    cost_limit: f32,
    lr_lambda: f32,
) -> f32 {
    (lambda_val + lr_lambda * (cost_mean - cost_limit)).max(0.0)
}

/// Conditional Value-at-Risk (CVaR_alpha) tail-risk metric (Chow et al. 2015).
pub fn cvar_risk_measure(costs: &Tensor, alpha: f32) -> Result<Tensor> {
    let mut c_vec = costs.flatten_all()?.to_vec1::<f32>()?;
    c_vec.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = c_vec.len();
    let k = ((alpha * n as f32).round() as usize).max(1);
    let sum_worst: f32 = c_vec[n - k..].iter().sum();

    Tensor::new(sum_worst / (k as f32), costs.device())
}

/// Control Barrier Function (CBF) quadratic program safety filter (Ames et al. 2019).
pub fn cbf_qp_safety_filter(
    u_nominal: &Tensor,
    grad_h: &Tensor,
    h_val: &Tensor,
    alpha_cbf: f32,
) -> Result<Tensor> {
    let un_vec = u_nominal.to_vec2::<f32>()?;
    let gh_vec = grad_h.to_vec2::<f32>()?;
    let hv_vec = h_val.flatten_all()?.to_vec1::<f32>()?;

    let b = un_vec.len();
    let d = un_vec[0].len();

    let mut u_safe = vec![vec![0.0f32; d]; b];

    for i in 0..b {
        let b_val = alpha_cbf * hv_vec[i];
        let mut a_dot_u = 0.0f32;
        let mut a_norm_sq = 0.0f32;

        for j in 0..d {
            a_dot_u += gh_vec[i][j] * un_vec[i][j];
            a_norm_sq += gh_vec[i][j] * gh_vec[i][j];
        }

        let c_viol = a_dot_u + b_val;
        let deficit = (-c_viol).max(0.0);
        let corr = deficit / (a_norm_sq + 1e-8);

        for j in 0..d {
            u_safe[i][j] = un_vec[i][j] + corr * gh_vec[i][j];
        }
    }

    Tensor::new(u_safe, u_nominal.device())
}
