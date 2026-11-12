//! Test Time Compute PRM Best of N and Mcts
//! in candle/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Process Reward Model (PRM) product aggregation over valid steps:
/// R_prod(i) = exp( sum_{s=1}^S m_{i,s} * ln(max(p_{i,s}, eps)) )
pub fn prm_product_aggregation(
    step_probs: &Tensor,
    step_mask: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let (b, s) = step_probs.dims2()?;
    let probs_vec = step_probs.to_vec2::<f32>()?;
    let mask_vec = step_mask.to_vec2::<f32>()?;

    let mut prod_out = Vec::with_capacity(b);
    for i in 0..b {
        let mut sum_log = 0.0f32;
        for j in 0..s {
            let p = probs_vec[i][j].max(eps);
            sum_log += mask_vec[i][j] * p.ln();
        }
        prod_out.push(sum_log.exp());
    }
    Tensor::from_vec(prod_out, b, step_probs.device())
}

/// Process Reward Model (PRM) min aggregation (weakest link principle):
/// R_min(i) = min_{s: m_{i,s} = 1} p_{i,s}
pub fn prm_min_aggregation(step_probs: &Tensor, step_mask: &Tensor) -> Result<Tensor> {
    let (b, s) = step_probs.dims2()?;
    let probs_vec = step_probs.to_vec2::<f32>()?;
    let mask_vec = step_mask.to_vec2::<f32>()?;

    let mut min_out = Vec::with_capacity(b);
    for i in 0..b {
        let mut min_val = 1.0f32;
        for j in 0..s {
            if mask_vec[i][j] > 0.5 {
                min_val = min_val.min(probs_vec[i][j]);
            }
        }
        min_out.push(min_val);
    }
    Tensor::from_vec(min_out, b, step_probs.device())
}

/// Combines ORM and PRM scores and identifies top-ranked candidate:
/// S_i = alpha * ORM_i + (1 - alpha) * PRM_i
pub fn best_of_n_rerank(
    orm_scores: &Tensor,
    prm_scores: &Tensor,
    alpha: f32,
) -> Result<(Tensor, Tensor)> {
    let orm = orm_scores.to_vec1::<f32>()?;
    let prm = prm_scores.to_vec1::<f32>()?;
    let n = orm.len();

    let mut comb = Vec::with_capacity(n);
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;

    for i in 0..n {
        let s = alpha * orm[i] + (1.0 - alpha) * prm[i];
        comb.push(s);
        if s > best_val {
            best_val = s;
            best_idx = i;
        }
    }

    let comb_t = Tensor::from_vec(comb, n, orm_scores.device())?;
    let idx_t = Tensor::from_vec(vec![best_idx as f32], 1, orm_scores.device())?;
    Ok((comb_t, idx_t))
}

/// AlphaZero Predictor Upper Confidence bounds for Trees (PUCT):
/// U(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt( sum_b N(s, b) ) / (1.0 + N(s, a))
pub fn puct_node_selection(
    q_values: &Tensor,
    priors: &Tensor,
    visit_counts: &Tensor,
    c_puct: f32,
) -> Result<(Tensor, Tensor)> {
    let q = q_values.to_vec1::<f32>()?;
    let p = priors.to_vec1::<f32>()?;
    let n = visit_counts.to_vec1::<f32>()?;
    let a = q.len();

    let total_visits: f32 = n.iter().sum();
    let sqrt_total = total_visits.sqrt();

    let mut puct = Vec::with_capacity(a);
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;

    for i in 0..a {
        let u = q[i] + c_puct * p[i] * sqrt_total / (1.0 + n[i]);
        puct.push(u);
        if u > best_val {
            best_val = u;
            best_idx = i;
        }
    }

    let puct_t = Tensor::from_vec(puct, a, q_values.device())?;
    let sel_t = Tensor::from_vec(vec![best_idx as f32], 1, q_values.device())?;
    Ok((puct_t, sel_t))
}

/// Online running average backpropagation update:
/// N'(s, a) = N(s, a) + 1
/// Q'(s, a) = Q(s, a) + (V - Q(s, a)) / N'(s, a)
pub fn mcts_backpropagation(
    q_values: &Tensor,
    visit_counts: &Tensor,
    action: usize,
    leaf_value: f32,
) -> Result<(Tensor, Tensor)> {
    let mut q = q_values.to_vec1::<f32>()?;
    let mut n = visit_counts.to_vec1::<f32>()?;
    let a = q.len();

    n[action] += 1.0;
    q[action] += (leaf_value - q[action]) / n[action];

    let q_t = Tensor::from_vec(q, a, q_values.device())?;
    let n_t = Tensor::from_vec(n, a, visit_counts.device())?;
    Ok((q_t, n_t))
}
