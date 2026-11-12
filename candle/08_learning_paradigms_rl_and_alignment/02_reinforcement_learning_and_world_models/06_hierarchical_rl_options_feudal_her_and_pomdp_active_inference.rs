//! Hierarchical RL Options Feudal Her and POMDP Active Inference
//! in candle/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Hindsight Experience Replay (HER) sparse binary goal achievement reward.
pub fn her_goal_relabeling(states: &Tensor, goals: &Tensor, threshold: f32) -> Result<Tensor> {
    let s = states.to_vec2::<f32>()?;
    let g = goals.to_vec2::<f32>()?;
    let b = s.len();
    let d = s[0].len();
    let mut out = Vec::with_capacity(b);

    for i in 0..b {
        let mut dist_sq = 0.0f32;
        for j in 0..d {
            let diff = s[i][j] - g[i][j];
            dist_sq += diff * diff;
        }
        out.push(if dist_sq.sqrt() <= threshold {
            1.0f32
        } else {
            0.0f32
        });
    }

    Tensor::from_vec(out, (b,), states.device())
}

/// FeUdal Networks cosine directional reward.
pub fn feudal_directional_reward(delta_s: &Tensor, subgoal_g: &Tensor, eps: f32) -> Result<Tensor> {
    let ds = delta_s.to_vec2::<f32>()?;
    let sg = subgoal_g.to_vec2::<f32>()?;
    let b = ds.len();
    let d = ds[0].len();
    let mut out = Vec::with_capacity(b);

    for i in 0..b {
        let mut dot = 0.0f32;
        let mut norm_ds_sq = 0.0f32;
        let mut norm_sg_sq = 0.0f32;
        for j in 0..d {
            dot += ds[i][j] * sg[i][j];
            norm_ds_sq += ds[i][j] * ds[i][j];
            norm_sg_sq += sg[i][j] * sg[i][j];
        }
        let denom = (norm_ds_sq.sqrt() * norm_sg_sq.sqrt()).max(eps);
        out.push(dot / denom);
    }

    Tensor::from_vec(out, (b,), delta_s.device())
}

/// Option-Critic termination gradient loss with deliberation cost.
pub fn option_critic_termination_loss(
    beta_omega: &Tensor,
    q_omega: &Tensor,
    v_s: &Tensor,
    deliberation_cost: f32,
) -> Result<Tensor> {
    let bo = beta_omega.to_vec1::<f32>()?;
    let qo = q_omega.to_vec1::<f32>()?;
    let vs = v_s.to_vec1::<f32>()?;
    let b = bo.len();
    let mut total = 0.0f32;

    for i in 0..b {
        total += bo[i] * (qo[i] - vs[i] + deliberation_cost);
    }

    Tensor::new(total / (b as f32), beta_omega.device())
}

/// Variational Free Energy in Active Inference / POMDP.
pub fn active_inference_free_energy(
    q_mean: &Tensor,
    q_log_std: &Tensor,
    p_mean: &Tensor,
    p_log_std: &Tensor,
    obs: &Tensor,
    recon: &Tensor,
    obs_noise_var: f32,
) -> Result<Tensor> {
    let qm = q_mean.to_vec2::<f32>()?;
    let qls = q_log_std.to_vec2::<f32>()?;
    let pm = p_mean.to_vec2::<f32>()?;
    let pls = p_log_std.to_vec2::<f32>()?;
    let ob = obs.to_vec2::<f32>()?;
    let rc = recon.to_vec2::<f32>()?;
    let b = qm.len();
    let d_s = qm[0].len();
    let d_o = ob[0].len();
    let mut fe_vec = Vec::with_capacity(b);
    let log_2pi_var = (2.0f32 * std::f32::consts::PI * obs_noise_var).ln();

    for i in 0..b {
        let mut kl = 0.0f32;
        for j in 0..d_s {
            let var_q = (2.0f32 * qls[i][j]).exp();
            let var_p = (2.0f32 * pls[i][j]).exp();
            let diff = qm[i][j] - pm[i][j];
            let el = (var_q + diff * diff) / var_p - 1.0f32 + 2.0f32 * (pls[i][j] - qls[i][j]);
            kl += 0.5f32 * el;
        }
        let mut nll = 0.0f32;
        for j in 0..d_o {
            let diff_o = ob[i][j] - rc[i][j];
            nll += 0.5f32 * ((diff_o * diff_o) / obs_noise_var + log_2pi_var);
        }
        fe_vec.push(kl + nll);
    }

    Tensor::from_vec(fe_vec, (b,), q_mean.device())
}
