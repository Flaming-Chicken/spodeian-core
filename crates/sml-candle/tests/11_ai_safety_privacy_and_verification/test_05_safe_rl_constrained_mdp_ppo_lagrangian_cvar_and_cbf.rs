use candle_core::{safetensors::load, Device, Result, Tensor};

use ai_candle::c11_ai_safety_privacy_and_verification::t05_safe_rl_constrained_mdp_ppo_lagrangian_cvar_and_cbf as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_05_safe_rl_constrained_mdp_ppo_lagrangian_cvar_and_cbf_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("11_ai_safety_privacy_and_verification/05_safe_rl_constrained_mdp_ppo_lagrangian_cvar_and_cbf.safetensors"),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. PPO-Lagrangian loss
    let ppo_loss = topic::ppo_lagrangian_loss(
        &w["logp"],
        &w["old_logp"],
        &w["adv_reward"],
        &w["adv_cost"],
        1.2,
        0.2,
    )?;
    common::assert_max_diff(&ppo_loss, &w["ppo_lag_ref"], 1e-5)?;

    // 3. Lagrangian multiplier step
    let lam_updated = topic::lagrangian_multiplier_step(1.2, 0.35, 0.20, 0.1);
    let lam_tensor = Tensor::new(lam_updated, &device)?;
    common::assert_max_diff(&lam_tensor, &w["lam_updated_ref"], 1e-5)?;

    // 4. CVaR risk measure
    let cvar_val = topic::cvar_risk_measure(&w["costs"], 0.25)?;
    common::assert_max_diff(&cvar_val, &w["cvar_ref"], 1e-5)?;

    // 5. CBF QP filter
    let u_safe = topic::cbf_qp_safety_filter(&w["u_nominal"], &w["grad_h"], &w["h_val"], 1.5)?;
    common::assert_max_diff(&u_safe, &w["u_safe_ref"], 1e-5)?;

    Ok(())
}
