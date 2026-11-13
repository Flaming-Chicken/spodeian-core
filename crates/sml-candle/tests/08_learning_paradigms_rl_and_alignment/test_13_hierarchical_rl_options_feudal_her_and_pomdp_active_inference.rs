use candle_core::{Device, Result, safetensors::load};

use ai_candle::c08_learning_paradigms_rl_and_alignment::t13_hierarchical_rl_options_feudal_her_and_pomdp_active_inference as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_13_hierarchical_rl_options_feudal_her_and_pomdp_active_inference_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/06_hierarchical_rl_options_feudal_her_and_pomdp_active_inference.safetensors",
        ),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. HER Goal Relabeling
    let her_rew = topic::her_goal_relabeling(&w["states"], &w["goals"], 0.1)?;
    common::assert_max_diff(&her_rew, &w["her_rewards_ref"], 1e-5)?;

    // 3. FeUdal Directional Reward
    let feudal_rew = topic::feudal_directional_reward(&w["delta_s"], &w["subgoal_g"], 1e-6)?;
    common::assert_max_diff(&feudal_rew, &w["feudal_rewards_ref"], 1e-5)?;

    // 4. Option-Critic Termination Loss
    let opt_loss =
        topic::option_critic_termination_loss(&w["beta_omega"], &w["q_omega"], &w["v_s"], 0.01)?;
    common::assert_max_diff(&opt_loss, &w["option_loss_ref"], 1e-5)?;

    // 5. Active Inference Free Energy
    let fe = topic::active_inference_free_energy(
        &w["q_mean"],
        &w["q_log_std"],
        &w["p_mean"],
        &w["p_log_std"],
        &w["obs"],
        &w["recon"],
        1.0,
    )?;
    common::assert_max_diff(&fe, &w["free_energy_ref"], 1e-5)?;

    Ok(())
}
