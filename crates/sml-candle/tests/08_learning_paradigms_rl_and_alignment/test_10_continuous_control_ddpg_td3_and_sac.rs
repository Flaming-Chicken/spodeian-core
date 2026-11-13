use candle_core::{Device, Result, safetensors::load};

use sml_candle::c08_learning_paradigms_rl_and_alignment::t10_continuous_control_ddpg_td3_and_sac as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_10_continuous_control_ddpg_td3_and_sac_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/03_continuous_control_ddpg_td3_and_sac.safetensors",
        ),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. TD3 Target Smoothing
    let smoothed = topic::td3_target_smoothing(&w["target_actions"], &w["noise"], 0.5, -1.0, 1.0)?;
    common::assert_max_diff(&smoothed, &w["smoothed_actions_ref"], 1e-5)?;

    // 3. TD3 Clipped Critic Target
    let targets = topic::td3_clipped_critic_target(
        &w["rewards"],
        &w["dones"],
        &w["q1_target"],
        &w["q2_target"],
        0.99,
    )?;
    common::assert_max_diff(&targets, &w["td3_targets_ref"], 1e-5)?;

    // 4. SAC Squashed Gaussian Policy
    let (actions, log_probs) =
        topic::sac_squashed_gaussian_log_prob(&w["u"], &w["mean"], &w["log_std"], 1e-6)?;
    common::assert_max_diff(&actions, &w["sac_actions_ref"], 1e-5)?;
    common::assert_max_diff(&log_probs, &w["sac_log_probs_ref"], 1e-5)?;

    // 5. SAC Temperature Loss
    let alpha_l = topic::sac_temperature_loss(&w["log_alpha"], &w["sac_log_probs_ref"], -3.0)?;
    common::assert_max_diff(&alpha_l, &w["alpha_loss_ref"], 1e-5)?;

    Ok(())
}
