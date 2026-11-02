//! Offline RL Imitation Cql IQL Gail and Decision Transformer
//! in candle/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Conservative Q-Learning (CQL) gap penalty:
/// L_cql = mean( logsumexp(Q(s, a)) - Q(s, a_data) ).
pub fn cql_penalty(q_values: &Tensor, dataset_actions: &Tensor) -> Result<Tensor> {
    let q = q_values.to_vec2::<f32>()?;
    let acts = dataset_actions.to_vec1::<i64>()?;
    let b = q.len();
    let mut sum_pen = 0.0f32;

    for i in 0..b {
        let max_val = q[i].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = q[i].iter().map(|&v| (v - max_val).exp()).sum();
        let logsumexp = max_val + sum_exp.ln();
        let data_q = q[i][acts[i] as usize];
        sum_pen += logsumexp - data_q;
    }

    Tensor::new(sum_pen / (b as f32), q_values.device())
}

/// Implicit Q-Learning (IQL) asymmetric expectile regression loss:
/// L_2^tau(u) = |tau - I(u < 0)| * u^2.
pub fn iql_expectile_loss(diff: &Tensor, tau: f32) -> Result<Tensor> {
    let d = diff.to_vec1::<f32>()?;
    let b = d.len();
    let mut sum_loss = 0.0f32;

    for &u in d.iter() {
        let weight = if u > 0.0f32 { tau } else { 1.0f32 - tau };
        sum_loss += weight * u * u;
    }

    Tensor::new(sum_loss / (b as f32), diff.device())
}

/// Advantage-Weighted Regression (AWR) actor loss in IQL.
pub fn iql_actor_loss(
    pred_action: &Tensor,
    target_action: &Tensor,
    adv: &Tensor,
    beta: f32,
    max_weight: f32,
) -> Result<Tensor> {
    let pa = pred_action.to_vec2::<f32>()?;
    let ta = target_action.to_vec2::<f32>()?;
    let a = adv.to_vec1::<f32>()?;
    let b = pa.len();
    let d = pa[0].len();
    let mut total_loss = 0.0f32;

    for i in 0..b {
        let w = (beta * a[i]).exp().min(max_weight);
        let mut sq_err = 0.0f32;
        for j in 0..d {
            let diff = pa[i][j] - ta[i][j];
            sq_err += diff * diff;
        }
        total_loss += w * sq_err;
    }

    Tensor::new(total_loss / (b as f32), pred_action.device())
}
