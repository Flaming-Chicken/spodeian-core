use candle_core::{safetensors::load, Device, Result};

use ai_candle::c07_domain_tasks::t13_robotics_embodied_ai_diffusion_policy_and_act as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_13_robotics_embodied_ai_diffusion_policy_and_act_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "07_domain_tasks/13_robotics_embodied_ai_diffusion_policy_and_act.safetensors",
        ),
        &device,
    )?;

    // Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 1. Temporal Ensemble Actions
    let ens = topic::temporal_ensemble_actions(&w["action_preds"], 0.1)?;
    common::assert_max_diff(&ens, &w["ensembled_ref"], 1e-5)?;

    // 2. Diffusion Policy DDPM Step
    let denoised = topic::diffusion_policy_ddpm_step(
        &w["action_t"],
        &w["noise_pred"],
        &w["noise_z"],
        0.99,
        0.85,
        0.05,
    )?;
    common::assert_max_diff(&denoised, &w["denoised_ref"], 1e-5)?;

    // 3. ACT CVAE KL Loss
    let kl = topic::act_cvae_kl_loss(&w["cvae_mu"], &w["cvae_logvar"])?;
    common::assert_max_diff(&kl, &w["kl_loss_ref"], 1e-5)?;

    // 4. Robot Action Safety Clamp
    let clamped = topic::robot_action_safety_clamp(&w["raw_actions"], &w["a_min"], &w["a_max"])?;
    common::assert_max_diff(&clamped, &w["clamped_ref"], 1e-5)?;

    Ok(())
}
