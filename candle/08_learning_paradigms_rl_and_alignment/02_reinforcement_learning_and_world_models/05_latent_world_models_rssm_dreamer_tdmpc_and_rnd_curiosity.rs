//! Latent World Models Rssm Dreamer Tdmpc and Rnd Curiosity
//! in candle/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Analytical KL divergence between diagonal Gaussians.
pub fn gaussian_kl_divergence(
    mu_q: &Tensor,
    log_std_q: &Tensor,
    mu_p: &Tensor,
    log_std_p: &Tensor,
) -> Result<Tensor> {
    let mq = mu_q.to_vec2::<f32>()?;
    let lsq = log_std_q.to_vec2::<f32>()?;
    let mp = mu_p.to_vec2::<f32>()?;
    let lsp = log_std_p.to_vec2::<f32>()?;
    let b = mq.len();
    let d = mq[0].len();
    let mut kl_vec = Vec::with_capacity(b);

    for i in 0..b {
        let mut sum_kl = 0.0f32;
        for j in 0..d {
            let var_q = (2.0f32 * lsq[i][j]).exp();
            let var_p = (2.0f32 * lsp[i][j]).exp();
            let diff = mq[i][j] - mp[i][j];
            let el = (var_q + diff * diff) / var_p - 1.0f32 + 2.0f32 * (lsp[i][j] - lsq[i][j]);
            sum_kl += 0.5f32 * el;
        }
        kl_vec.push(sum_kl);
    }

    Tensor::from_vec(kl_vec, (b,), mu_q.device())
}

/// Dreamer / RSSM KL balancing objective.
pub fn rssm_kl_balancing(
    mu_q: &Tensor,
    log_std_q: &Tensor,
    mu_p: &Tensor,
    log_std_p: &Tensor,
    beta_prior: f32,
    beta_post: f32,
) -> Result<Tensor> {
    let kl = gaussian_kl_divergence(mu_q, log_std_q, mu_p, log_std_p)?;
    let mean_kl = kl.mean_all()?.to_scalar::<f32>()?;
    Tensor::new(mean_kl * (beta_prior + beta_post), mu_q.device())
}

/// Random Network Distillation (RND) intrinsic curiosity reward.
pub fn rnd_intrinsic_reward(
    target_features: &Tensor,
    predictor_features: &Tensor,
) -> Result<Tensor> {
    let tf = target_features.to_vec2::<f32>()?;
    let pf = predictor_features.to_vec2::<f32>()?;
    let b = tf.len();
    let d = tf[0].len();
    let mut rewards = Vec::with_capacity(b);

    for i in 0..b {
        let mut sum_sq = 0.0f32;
        for j in 0..d {
            let diff = pf[i][j] - tf[i][j];
            sum_sq += diff * diff;
        }
        rewards.push(sum_sq / (d as f32));
    }

    Tensor::from_vec(rewards, (b,), target_features.device())
}

/// TD-MPC2 latent imagination rollout return.
pub fn tdmpc_latent_rollout_return(
    latent_rewards: &Tensor,
    terminal_q: &Tensor,
    gamma: f32,
) -> Result<Tensor> {
    let lr = latent_rewards.to_vec2::<f32>()?;
    let tq = terminal_q.to_vec1::<f32>()?;
    let b = lr.len();
    let h = lr[0].len();
    let mut returns = Vec::with_capacity(b);

    for i in 0..b {
        let mut discounted_r = 0.0f32;
        let mut discount = 1.0f32;
        for t in 0..h {
            discounted_r += lr[i][t] * discount;
            discount *= gamma;
        }
        returns.push(discounted_r + discount * tq[i]);
    }

    Tensor::from_vec(returns, (b,), latent_rewards.device())
}
