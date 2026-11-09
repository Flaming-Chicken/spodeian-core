use candle_core::{safetensors::load, Device, Result};

use ai_candle::c10_xai_and_ai_analysis_methods::t03_xai_concepts_tcav_cbm_counterfactuals_dice_and_global_ale as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_03_xai_concepts_tcav_cbm_counterfactuals_dice_and_global_ale_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("10_xai_and_ai_analysis_methods/03_xai_concepts_tcav_cbm_counterfactuals_dice_and_global_ale.safetensors"),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. TCAV
    let (derivs, score) = topic::tcav_directional_derivatives(&w["tcav_grads"], &w["tcav_cav"])?;
    common::assert_max_diff(&derivs, &w["tcav_derivs_ref"], 1e-5)?;
    common::assert_max_diff(&score, &w["tcav_score_ref"], 1e-5)?;

    // 3. CBM
    let (c_int, y_pred) = topic::cbm_test_time_intervention(
        &w["cbm_pred_c"],
        &w["cbm_true_c"],
        &w["cbm_mask"],
        &w["cbm_w"],
        &w["cbm_b"],
    )?;
    common::assert_max_diff(&c_int, &w["cbm_c_int_ref"], 1e-5)?;
    common::assert_max_diff(&y_pred, &w["cbm_y_ref"], 1e-5)?;

    // 4. DiCE
    let div = topic::dice_dpp_diversity(&w["dice_cfs"])?;
    common::assert_max_diff(&div, &w["dice_div_ref"], 1e-5)?;

    // 5. ALE
    let ale = topic::ale_accumulated_local_effects(&w["ale_diffs"])?;
    common::assert_max_diff(&ale, &w["ale_ref"], 1e-5)?;

    Ok(())
}
