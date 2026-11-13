use candle_core::{Device, Result, safetensors::load};

use sml_candle::c08_learning_paradigms_rl_and_alignment::t17_rlhf_reward_modeling_and_ppo_vs_rloo as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_17_rlhf_reward_modeling_and_ppo_vs_rloo_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/02_rlhf_reward_modeling_and_ppo_vs_rloo.safetensors",
        ),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Bradley-Terry Loss
    let bt_loss = topic::bradley_terry_reward_loss(&w["rewards_chosen"], &w["rewards_rejected"])?;
    common::assert_max_diff(&bt_loss, &w["bt_loss_ref"], 1e-5)?;

    // 3. RLHF Token Rewards
    let rew = topic::rlhf_token_level_rewards(
        &w["log_probs_policy"],
        &w["log_probs_ref"],
        &w["terminal_scores"],
        0.05,
    )?;
    common::assert_max_diff(&rew, &w["rlhf_rewards_ref"], 1e-5)?;

    // 4. RLOO Advantages
    let adv = topic::rloo_leave_one_out_advantages(&w["scores"])?;
    common::assert_max_diff(&adv, &w["rloo_adv_ref"], 1e-5)?;

    // 5. RLOO Loss
    let loss = topic::rloo_policy_gradient_loss(&w["rloo_adv_ref"], &w["log_probs_sum"])?;
    common::assert_max_diff(&loss, &w["rloo_loss_ref"], 1e-5)?;

    Ok(())
}
