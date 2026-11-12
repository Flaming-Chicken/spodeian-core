//! Candle implementation for Category 01 Topic 10:
//! Levels of Decision Making: Bayes Risk, Bandits, MDP, POMDP, and Games.

use candle_core::{DType, Result, Tensor};
use candle_nn::{Linear, Module};

/// Level 0: Static Bayes decision rule with Chow's rejection option.
pub fn bayes_decision_rule(
    probs: &Tensor,
    cost_matrix: &Tensor,
    c_reject: f32,
) -> Result<(Tensor, Tensor, Tensor)> {
    let expected_risk = probs.matmul(&cost_matrix.t()?)?;
    let best_actions = expected_risk.argmin(1)?.to_dtype(DType::F32)?;
    let min_risk = expected_risk.min(1)?;
    let reject_mask = min_risk.gt(c_reject as f64)?.to_dtype(DType::F32)?;
    Ok((expected_risk, best_actions, reject_mask))
}

/// Level 1: Contextual bandits off-policy evaluation (IPS and Doubly Robust).
pub fn bandits_ope(
    target_probs: &Tensor,
    logged_propensities: &Tensor,
    rewards: &Tensor,
    baseline_q: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let ips_weights = (target_probs / logged_propensities)?;
    let diff = (rewards - baseline_q)?;
    let weighted_diff = (&ips_weights * &diff)?;
    let doubly_robust = (baseline_q + &weighted_diff)?;
    Ok((ips_weights, doubly_robust))
}

/// Level 2: Closed-loop MDP Bellman optimality backup operator.
pub fn mdp_bellman_operator(
    v_current: &Tensor,
    transitions: &Tensor,
    rewards: &Tensor,
    gamma: f64,
) -> Result<(Tensor, Tensor)> {
    let dims = transitions.dims();
    let num_states = dims[0];
    let num_actions = dims[1];
    let next_states = dims[2];

    let t_flat = transitions.reshape((num_states * num_actions, next_states))?;
    let v_col = v_current.reshape((next_states, 1))?;
    let future_flat = t_flat.matmul(&v_col)?;
    let future_v = future_flat.reshape((num_states, num_actions))?;

    let gamma_future_v = (future_v * gamma)?;
    let q_values = (rewards + &gamma_future_v)?;
    let v_next = q_values.max(1)?;
    Ok((q_values, v_next))
}

/// Level 3: Bayesian belief-state filtering for POMDPs.
pub fn pomdp_belief_filter(
    prior_belief: &Tensor,
    transition: &Tensor,
    observation_likelihood: &Tensor,
) -> Result<Tensor> {
    let num_states = prior_belief.dims()[0];
    let b_row = prior_belief.reshape((1, num_states))?;
    let b_pred = b_row.matmul(transition)?.reshape(num_states)?;
    let b_unnorm = (observation_likelihood * &b_pred)?;
    let sum_unnorm = b_unnorm.sum_all()?.to_scalar::<f32>()? as f64;
    let posterior = (&b_unnorm / sum_unnorm)?;
    Ok(posterior)
}

/// Level 5: Differentiable 2-bidder auction mechanism.
pub fn differentiable_auction(bids: &Tensor, tau: f64) -> Result<(Tensor, Tensor)> {
    let scaled = (bids / tau)?;
    let allocations = candle_nn::ops::softmax(&scaled, 1)?;
    let min_bid = bids.min_keepdim(1)?;
    let payments = allocations.broadcast_mul(&min_bid)?;
    Ok((allocations, payments))
}

/// Top-level module encapsulating linear anchor.
pub struct DecisionHierarchyModule {
    pub linear: Linear,
}

impl DecisionHierarchyModule {
    pub fn new(weight: Tensor, bias: Tensor) -> Self {
        Self {
            linear: Linear::new(weight, Some(bias)),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.linear.forward(x)
    }
}
