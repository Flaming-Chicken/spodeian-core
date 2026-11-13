use candle_core::{Device, Result, safetensors::load};

use ai_candle::c09_causal_ai_and_evolutionary_search::t07_causal_representation_identifiability_ivae_synthetic_control_and_symbolic_sr as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_07_causal_representation_identifiability_ivae_synthetic_control_and_symbolic_sr_parity()
-> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "09_causal_ai_and_evolutionary_search/07_causal_representation_identifiability_ivae_synthetic_control_and_symbolic_sr.safetensors",
        ),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. iVAE conditional Gaussian KL divergence
    let kl = topic::ivae_kl_divergence(&w["mu_z"], &w["logvar_z"], &w["mu_u"], &w["logvar_u"])?;
    common::assert_max_diff(&kl, &w["ivae_kl_ref"], 1e-5)?;

    // 3. Simplex projection
    let w_simp = topic::project_simplex(&w["sc_v"])?;
    common::assert_max_diff(&w_simp, &w["sc_w_simplex_ref"], 1e-5)?;

    // 4. Synthetic control trajectory & treatment effect
    let (synth, tau) =
        topic::synthetic_control_effect(&w["sc_y1_post"], &w["sc_y0_post"], &w_simp)?;
    common::assert_max_diff(&synth, &w["sc_y1_synth_ref"], 1e-5)?;
    common::assert_max_diff(&tau, &w["sc_tau_ref"], 1e-5)?;

    // 5. Symbolic regression Pareto loss
    let pareto = topic::symbolic_pareto_loss(
        &w["sr_y_true"],
        &w["sr_y_pred"],
        &w["sr_complexity"],
        &w["sr_lambda"],
    )?;
    common::assert_max_diff(&pareto, &w["sr_pareto_loss_ref"], 1e-5)?;

    Ok(())
}
