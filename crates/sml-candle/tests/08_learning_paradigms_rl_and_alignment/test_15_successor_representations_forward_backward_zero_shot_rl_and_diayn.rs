use candle_core::{Device, Result, safetensors::load};

use ai_candle::c08_learning_paradigms_rl_and_alignment::t15_successor_representations_forward_backward_zero_shot_rl_and_diayn as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_15_successor_representations_forward_backward_zero_shot_rl_and_diayn_parity() -> Result<()>
{
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/08_successor_representations_forward_backward_zero_shot_rl_and_diayn.safetensors",
        ),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Successor Features TD Error
    let sf_td = topic::successor_features_td_error(
        &w["psi_current"],
        &w["features"],
        &w["psi_next"],
        &w["dones"],
        0.99,
    )?;
    common::assert_max_diff(&sf_td, &w["sf_td_error_ref"], 1e-5)?;

    // 3. GPI Transfer Value
    let gpi_val = topic::gpi_transfer_value(&w["psi_policies"], &w["reward_weights"])?;
    common::assert_max_diff(&gpi_val, &w["gpi_values_ref"], 1e-5)?;

    // 4. FB Zero-Shot Value
    let fb_val = topic::fb_zero_shot_value(&w["forward_map"], &w["task_embedding"])?;
    common::assert_max_diff(&fb_val, &w["fb_values_ref"], 1e-5)?;

    // 5. DIAYN Skill Reward and Loss
    let (diayn_rew, diayn_loss) =
        topic::diayn_skill_reward_and_loss(&w["logits"], &w["skill_labels"])?;
    common::assert_max_diff(&diayn_rew, &w["diayn_rewards_ref"], 1e-5)?;
    common::assert_max_diff(&diayn_loss, &w["diayn_loss_ref"], 1e-5)?;

    Ok(())
}
