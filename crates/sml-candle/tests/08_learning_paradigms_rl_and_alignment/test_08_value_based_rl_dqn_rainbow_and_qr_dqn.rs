use candle_core::{safetensors::load, Device, Result};

use ai_candle::c08_learning_paradigms_rl_and_alignment::t08_value_based_rl_dqn_rainbow_and_qr_dqn as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_08_value_based_rl_dqn_rainbow_and_qr_dqn_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/01_value_based_rl_dqn_rainbow_and_qr_dqn.safetensors"),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Dueling Q
    let dueling_q = topic::dueling_q_values(&w["value"], &w["advantage"])?;
    common::assert_max_diff(&dueling_q, &w["dueling_q_ref"], 1e-5)?;

    // 3. Double DQN targets
    let ddqn_targets = topic::double_dqn_targets(
        &w["rewards"],
        &w["dones"],
        &w["q_next_online"],
        &w["q_next_target"],
        0.99,
    )?;
    common::assert_max_diff(&ddqn_targets, &w["ddqn_targets_ref"], 1e-5)?;

    // 4. PER loss
    let per_loss = topic::per_loss(&w["q_eval"], &w["targets"], &w["is_weights"])?;
    common::assert_max_diff(&per_loss, &w["per_loss_ref"], 1e-5)?;

    // 5. QR-DQN loss
    let qr_loss = topic::quantile_huber_loss(
        &w["current_quantiles"],
        &w["target_quantiles"],
        &w["tau"],
        1.0,
    )?;
    common::assert_max_diff(&qr_loss, &w["qr_loss_ref"], 1e-5)?;

    Ok(())
}
