use candle_core::{safetensors::load, Device, Result};

use ai_candle::c08_learning_paradigms_rl_and_alignment::t09_policy_gradient_reinforce_a2c_ppo_and_gflownets as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_09_policy_gradient_reinforce_a2c_ppo_and_gflownets_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/02_policy_gradient_reinforce_a2c_ppo_and_gflownets.safetensors"),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. PPO Clip loss
    let ppo_l = topic::ppo_clip_loss(&w["log_probs"], &w["old_log_probs"], &w["advantages"], 0.2)?;
    common::assert_max_diff(&ppo_l, &w["ppo_loss_ref"], 1e-5)?;

    // 3. GAE
    let next_val = w["next_value"].to_scalar::<f32>()?;
    let gae_adv = topic::compute_gae(
        &w["rewards"],
        &w["values"],
        &w["dones"],
        next_val,
        0.99,
        0.95,
    )?;
    common::assert_max_diff(&gae_adv, &w["gae_advantages_ref"], 1e-5)?;

    // 4. GFlowNet TB loss
    let tb_l = topic::gflownet_trajectory_balance_loss(
        &w["log_z"],
        &w["log_pf"],
        &w["log_pb"],
        &w["reward"],
    )?;
    common::assert_max_diff(&tb_l, &w["tb_loss_ref"], 1e-5)?;

    Ok(())
}
