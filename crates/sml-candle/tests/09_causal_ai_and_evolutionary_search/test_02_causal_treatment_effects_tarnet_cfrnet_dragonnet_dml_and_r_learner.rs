use candle_core::{Device, Result, safetensors::load};

use ai_candle::c09_causal_ai_and_evolutionary_search::t02_causal_treatment_effects_tarnet_cfrnet_dragonnet_dml_and_r_learner as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_02_causal_treatment_effects_tarnet_cfrnet_dragonnet_dml_and_r_learner_parity() -> Result<()>
{
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "09_causal_ai_and_evolutionary_search/02_causal_treatment_effects_tarnet_cfrnet_dragonnet_dml_and_r_learner.safetensors",
        ),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. TARNet forward
    let (phi, y0, y1, tau) = topic::tarnet_forward(
        &w["test_input"],
        &w["phi_w"],
        &w["phi_b"],
        &w["h0_w"],
        &w["h0_b"],
        &w["h1_w"],
        &w["h1_b"],
    )?;
    common::assert_max_diff(&y0, &w["y0_ref"], 1e-5)?;
    common::assert_max_diff(&y1, &w["y1_ref"], 1e-5)?;
    common::assert_max_diff(&tau, &w["tau_ref"], 1e-5)?;

    // 3. CFRNet MMD loss
    let mmd = topic::cfrnet_mmd_loss(&phi, &w["t_treatment"])?;
    common::assert_max_diff(&mmd, &w["mmd_loss_ref"], 1e-5)?;

    // 4. Dragonnet propensity head
    let (prop, bce) = topic::dragonnet_propensity_loss(
        &phi,
        &w["t_treatment"],
        &w["w_prop"],
        &w["b_prop"],
        1e-8,
    )?;
    common::assert_max_diff(&prop, &w["propensity_ref"], 1e-5)?;
    common::assert_max_diff(&bce, &w["prop_loss_ref"], 1e-5)?;

    // 5. DML Robinson CATE estimator
    let dml_tau =
        topic::dml_cate_estimate(&w["y_outcome"], &w["t_treatment"], &w["m_hat"], &w["e_hat"])?;
    common::assert_max_diff(&dml_tau, &w["dml_tau_ref"], 1e-5)?;

    Ok(())
}
