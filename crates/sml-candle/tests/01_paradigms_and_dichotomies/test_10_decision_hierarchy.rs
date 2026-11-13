//! Integration test for Category 01 Topic 10: Levels of Decision Making (Candle Track).

use sml_candle::c01_paradigms_and_dichotomies::t10_levels_of_decision_making_bayes_risk_bandits_mdp_pomdp_and_games::{
    bandits_ope, bayes_decision_rule, differentiable_auction, mdp_bellman_operator,
    pomdp_belief_filter,
};
use candle_core::{safetensors::load, Device, Result};

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_10_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "01_paradigms_and_dichotomies/10_levels_of_decision_making_bayes_risk_bandits_mdp_pomdp_and_games.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let test_in = tensors.get("test_input").expect("test_input");
    let test_out = tensors.get("test_output").expect("test_output");
    let weight = tensors.get("layers.weight").expect("layers.weight");
    let bias = tensors.get("layers.bias").expect("layers.bias");

    // 1. Canonical Linear Anchor Parity
    let linear_pred = test_in.matmul(weight)?.broadcast_add(bias)?;
    let diff_linear = linear_pred
        .sub(test_out)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_linear < 1e-5,
        "Linear anchor parity error {:.2e} exceeds 1e-5",
        diff_linear
    );

    // 2. Level 0: Bayes Risk & Chow Rejection
    let l0_probs = tensors.get("layers.l0_probs").expect("l0_probs");
    let l0_cost = tensors
        .get("layers.l0_cost_matrix")
        .expect("l0_cost_matrix");
    let l0_risk_gold = tensors
        .get("layers.l0_expected_risk")
        .expect("l0_expected_risk");
    let l0_acts_gold = tensors
        .get("layers.l0_best_actions")
        .expect("l0_best_actions");
    let l0_rej_gold = tensors
        .get("layers.l0_reject_mask")
        .expect("l0_reject_mask");

    let (l0_risk_calc, l0_acts_calc, l0_rej_calc) = bayes_decision_rule(l0_probs, l0_cost, 0.5f32)?;
    let diff_risk = l0_risk_calc
        .sub(l0_risk_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_risk < 1e-5,
        "L0 risk error {:.2e} exceeds 1e-5",
        diff_risk
    );

    let diff_acts = l0_acts_calc
        .sub(l0_acts_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_acts < 1e-5,
        "L0 acts error {:.2e} exceeds 1e-5",
        diff_acts
    );

    let diff_rej = l0_rej_calc
        .sub(l0_rej_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_rej < 1e-5,
        "L0 reject error {:.2e} exceeds 1e-5",
        diff_rej
    );

    // 3. Level 1: Bandits OPE
    let l1_tp = tensors.get("layers.l1_target_policy_probs").expect("l1_tp");
    let l1_lp = tensors.get("layers.l1_logged_propensities").expect("l1_lp");
    let l1_rew = tensors.get("layers.l1_rewards").expect("l1_rew");
    let l1_q = tensors.get("layers.l1_baseline_q").expect("l1_q");
    let l1_w_gold = tensors
        .get("layers.l1_ips_weights")
        .expect("l1_ips_weights");
    let l1_dr_gold = tensors
        .get("layers.l1_doubly_robust_estimates")
        .expect("l1_doubly_robust");

    let (l1_w_calc, l1_dr_calc) = bandits_ope(l1_tp, l1_lp, l1_rew, l1_q)?;
    let diff_w = l1_w_calc
        .sub(l1_w_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_w < 1e-5, "L1 weight error {:.2e} exceeds 1e-5", diff_w);

    let diff_dr = l1_dr_calc
        .sub(l1_dr_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_dr < 1e-5, "L1 DR error {:.2e} exceeds 1e-5", diff_dr);

    // 4. Level 2: MDP Bellman Operator
    let l2_v_init = tensors.get("layers.l2_v_init").expect("l2_v_init");
    let l2_trans = tensors
        .get("layers.l2_transitions")
        .expect("l2_transitions");
    let l2_rew = tensors.get("layers.l2_rewards").expect("l2_rewards");
    let l2_q_gold = tensors.get("layers.l2_q_values").expect("l2_q_values");
    let l2_v_gold = tensors.get("layers.l2_v_backup").expect("l2_v_backup");

    let (l2_q_calc, l2_v_calc) = mdp_bellman_operator(l2_v_init, l2_trans, l2_rew, 0.9)?;
    let diff_q = l2_q_calc
        .sub(l2_q_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_q < 1e-5, "L2 Q error {:.2e} exceeds 1e-5", diff_q);

    let diff_v = l2_v_calc
        .sub(l2_v_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_v < 1e-5, "L2 V error {:.2e} exceeds 1e-5", diff_v);

    // 5. Level 3: POMDP Belief Filter
    let l3_prior = tensors.get("layers.l3_prior_belief").expect("l3_prior");
    let l3_trans = tensors.get("layers.l3_transition").expect("l3_trans");
    let l3_obs = tensors
        .get("layers.l3_observation_likelihood")
        .expect("l3_obs");
    let l3_post_gold = tensors.get("layers.l3_posterior_belief").expect("l3_post");

    let l3_post_calc = pomdp_belief_filter(l3_prior, l3_trans, l3_obs)?;
    let diff_post = l3_post_calc
        .sub(l3_post_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_post < 1e-5,
        "L3 post error {:.2e} exceeds 1e-5",
        diff_post
    );

    // 6. Level 5: Differentiable Auction
    let l5_bids = tensors.get("layers.l5_bids").expect("l5_bids");
    let l5_alloc_gold = tensors
        .get("layers.l5_allocations")
        .expect("l5_allocations");
    let l5_pay_gold = tensors.get("layers.l5_payments").expect("l5_payments");

    let (l5_alloc_calc, l5_pay_calc) = differentiable_auction(l5_bids, 0.5)?;
    let diff_alloc = l5_alloc_calc
        .sub(l5_alloc_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_alloc < 1e-5,
        "L5 alloc error {:.2e} exceeds 1e-5",
        diff_alloc
    );

    let diff_pay = l5_pay_calc
        .sub(l5_pay_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_pay < 1e-5,
        "L5 pay error {:.2e} exceeds 1e-5",
        diff_pay
    );

    Ok(())
}
