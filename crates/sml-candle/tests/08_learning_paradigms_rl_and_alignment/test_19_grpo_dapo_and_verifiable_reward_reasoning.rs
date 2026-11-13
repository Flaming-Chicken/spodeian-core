use candle_core::{Device, Result, safetensors::load};

use sml_candle::c08_learning_paradigms_rl_and_alignment::t19_grpo_dapo_and_verifiable_reward_reasoning as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_19_grpo_dapo_and_verifiable_reward_reasoning_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/04_grpo_dapo_and_verifiable_reward_reasoning.safetensors",
        ),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Composite Verifiable Rewards
    let rewards =
        topic::composite_verifiable_reward(&w["accuracy_flags"], &w["format_flags"], 1.0, 0.5)?;
    common::assert_max_diff(&rewards, &w["rewards_ref"], 1e-5)?;

    // 3. GRPO Group Advantages
    let adv = topic::grpo_group_advantages(&w["rewards_ref"], 1e-6)?;
    common::assert_max_diff(&adv, &w["advantages_ref"], 1e-5)?;

    // 4. K3 KL Divergence
    let kl = topic::k3_kl_divergence(&w["log_probs_policy"], &w["log_probs_ref"])?;
    common::assert_max_diff(&kl, &w["kl_div_ref"], 1e-5)?;

    // 5. GRPO Clipped Loss
    let loss = topic::grpo_clipped_loss(
        &w["log_probs_policy"],
        &w["log_probs_old"],
        &w["advantages_ref"],
        &w["kl_div_ref"],
        0.2,
        0.04,
    )?;
    common::assert_max_diff(&loss, &w["grpo_loss_ref"], 1e-5)?;

    Ok(())
}
