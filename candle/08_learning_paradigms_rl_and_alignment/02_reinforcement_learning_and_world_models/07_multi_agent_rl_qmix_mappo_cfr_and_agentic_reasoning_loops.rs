//! Multi Agent RL QMIX Mappo Cfr and Agentic Reasoning Loops
//! in candle/08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// QMIX Monotonic Hypernetwork Mixing operator with absolute non-negative weights.
pub fn qmix_monotonic_mixing(
    agent_qs: &Tensor,
    w1_raw: &Tensor,
    b1: &Tensor,
    w2_raw: &Tensor,
    b2: &Tensor,
) -> Result<Tensor> {
    let qs = agent_qs.to_vec3::<f32>()?;
    let w1 = w1_raw.to_vec3::<f32>()?;
    let b1_vec = b1.to_vec3::<f32>()?;
    let w2 = w2_raw.to_vec3::<f32>()?;
    let b2_vec = b2.to_vec3::<f32>()?;

    let b = qs.len();
    let n = qs[0][0].len();
    let h_dim = w1[0][0].len();

    let mut q_tot_vec = Vec::with_capacity(b);

    for batch in 0..b {
        let mut hidden = vec![0.0f32; h_dim];
        for h in 0..h_dim {
            let mut sum_val = b1_vec[batch][0][h];
            for i in 0..n {
                sum_val += qs[batch][0][i] * w1[batch][i][h].abs();
            }
            hidden[h] = if sum_val > 0.0 {
                sum_val
            } else {
                sum_val.exp() - 1.0
            };
        }

        let mut q_tot = b2_vec[batch][0][0];
        for h in 0..h_dim {
            q_tot += hidden[h] * w2[batch][h][0].abs();
        }
        q_tot_vec.push(q_tot);
    }

    Tensor::from_vec(q_tot_vec, (b,), agent_qs.device())
}

/// MAPPO Centralized Generalized Advantage Estimation (GAE).
pub fn mappo_centralized_gae(
    values: &Tensor,
    rewards: &Tensor,
    dones: &Tensor,
    next_value: &Tensor,
    gamma: f32,
    gae_lambda: f32,
) -> Result<Tensor> {
    let val = values.to_vec2::<f32>()?;
    let rew = rewards.to_vec2::<f32>()?;
    let don = dones.to_vec2::<f32>()?;
    let next_v = next_value.to_vec1::<f32>()?;

    let b = val.len();
    let t = val[0].len();
    let mut adv_flat = vec![0.0f32; b * t];

    for batch in 0..b {
        let mut gae = 0.0f32;
        for step in (0..t).rev() {
            let nv = if step == t - 1 {
                next_v[batch]
            } else {
                val[batch][step + 1]
            };
            let non_terminal = 1.0f32 - don[batch][step];
            let delta = rew[batch][step] + gamma * nv * non_terminal - val[batch][step];
            gae = delta + gamma * gae_lambda * non_terminal * gae;
            adv_flat[batch * t + step] = gae;
        }
    }

    Tensor::from_vec(adv_flat, (b, t), values.device())
}

/// CFR Regret Matching for imperfect information game strategy updating.
pub fn cfr_regret_matching(cumulative_regrets: &Tensor, eps: f32) -> Result<Tensor> {
    let reg = cumulative_regrets.to_vec2::<f32>()?;
    let b = reg.len();
    let a_dim = reg[0].len();
    let mut strat_flat = Vec::with_capacity(b * a_dim);

    for batch in 0..b {
        let mut sum_pos = 0.0f32;
        for a in 0..a_dim {
            if reg[batch][a] > 0.0 {
                sum_pos += reg[batch][a];
            }
        }

        if sum_pos > eps {
            for a in 0..a_dim {
                let pos = reg[batch][a].max(0.0f32);
                strat_flat.push(pos / (sum_pos + eps));
            }
        } else {
            let uniform = 1.0f32 / (a_dim as f32);
            for _ in 0..a_dim {
                strat_flat.push(uniform);
            }
        }
    }

    Tensor::from_vec(strat_flat, (b, a_dim), cumulative_regrets.device())
}

/// Multi-agent reasoning / debate consensus metric across agent thoughts.
pub fn agentic_consensus_metric(embeddings: &Tensor, eps: f32) -> Result<Tensor> {
    let emb = embeddings.to_vec3::<f32>()?;
    let b = emb.len();
    let n = emb[0].len();
    let d = emb[0][0].len();

    let mut consensus_vec = Vec::with_capacity(b);

    for batch in 0..b {
        // Compute normalized vectors
        let mut normed = vec![vec![0.0f32; d]; n];
        for i in 0..n {
            let mut norm_sq = 0.0f32;
            for k in 0..d {
                norm_sq += emb[batch][i][k] * emb[batch][i][k];
            }
            let norm = norm_sq.sqrt().max(eps);
            for k in 0..d {
                normed[i][k] = emb[batch][i][k] / norm;
            }
        }

        // Pairwise cosine similarity for i != j
        let mut sum_sim = 0.0f32;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let mut dot = 0.0f32;
                    for k in 0..d {
                        dot += normed[i][k] * normed[j][k];
                    }
                    sum_sim += dot;
                }
            }
        }

        let num_pairs = (n * (n - 1)) as f32;
        consensus_vec.push(sum_sim / num_pairs);
    }

    Tensor::from_vec(consensus_vec, (b,), embeddings.device())
}
