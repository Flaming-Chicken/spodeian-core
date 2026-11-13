use candle_core::{Device, Result, safetensors::load};

use sml_candle::c08_learning_paradigms_rl_and_alignment::t12_latent_world_models_rssm_dreamer_tdmpc_and_rnd_curiosity as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_12_latent_world_models_rssm_dreamer_tdmpc_and_rnd_curiosity_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "08_learning_paradigms_rl_and_alignment/02_reinforcement_learning_and_world_models/05_latent_world_models_rssm_dreamer_tdmpc_and_rnd_curiosity.safetensors",
        ),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Gaussian KL divergence
    let kl_div =
        topic::gaussian_kl_divergence(&w["mu_q"], &w["log_std_q"], &w["mu_p"], &w["log_std_p"])?;
    common::assert_max_diff(&kl_div, &w["kl_div_ref"], 1e-5)?;

    // 3. RSSM KL Balancing
    let kl_balanced = topic::rssm_kl_balancing(
        &w["mu_q"],
        &w["log_std_q"],
        &w["mu_p"],
        &w["log_std_p"],
        0.8,
        0.2,
    )?;
    common::assert_max_diff(&kl_balanced, &w["kl_balanced_ref"], 1e-5)?;

    // 4. RND Curiosity
    let rnd_rew = topic::rnd_intrinsic_reward(&w["target_features"], &w["predictor_features"])?;
    common::assert_max_diff(&rnd_rew, &w["rnd_rewards_ref"], 1e-5)?;

    // 5. TD-MPC2 Rollout Return
    let tdmpc_ret =
        topic::tdmpc_latent_rollout_return(&w["latent_rewards"], &w["terminal_q"], 0.99)?;
    common::assert_max_diff(&tdmpc_ret, &w["tdmpc_return_ref"], 1e-5)?;

    Ok(())
}
