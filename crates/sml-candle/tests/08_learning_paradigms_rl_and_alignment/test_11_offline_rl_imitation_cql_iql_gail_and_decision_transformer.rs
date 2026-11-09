use candle_core::{safetensors::load, Device, Result};

use ai_candle::c08_learning_paradigms_rl_and_alignment::t11_offline_rl_imitation_cql_iql_gail_and_decision_transformer as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_11_offline_rl_imitation_cql_iql_gail_and_decision_transformer_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/04_offline_rl_imitation_cql_iql_gail_and_decision_transformer.safetensors"),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. CQL Penalty
    let cql_pen = topic::cql_penalty(&w["q_values"], &w["dataset_actions"])?;
    common::assert_max_diff(&cql_pen, &w["cql_penalty_ref"], 1e-5)?;

    // 3. IQL Expectile Loss
    let iql_exp = topic::iql_expectile_loss(&w["diff"], 0.7)?;
    common::assert_max_diff(&iql_exp, &w["iql_expectile_ref"], 1e-5)?;

    // 4. IQL Actor Loss
    let iql_act = topic::iql_actor_loss(
        &w["pred_action"],
        &w["target_action"],
        &w["adv"],
        3.0,
        100.0,
    )?;
    common::assert_max_diff(&iql_act, &w["iql_actor_ref"], 1e-5)?;

    Ok(())
}
