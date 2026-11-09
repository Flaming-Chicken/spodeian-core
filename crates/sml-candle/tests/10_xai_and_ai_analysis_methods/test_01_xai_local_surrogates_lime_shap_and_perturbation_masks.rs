use candle_core::{safetensors::load, Device, Result};

use ai_candle::c10_xai_and_ai_analysis_methods::t01_xai_local_surrogates_lime_shap_and_perturbation_masks as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_01_xai_local_surrogates_lime_shap_and_perturbation_masks_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("10_xai_and_ai_analysis_methods/01_xai_local_surrogates_lime_shap_and_perturbation_masks.safetensors"),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. LIME weighted regression
    let lime_w = topic::lime_weighted_regression(&w["lime_z"], &w["lime_y"], &w["lime_pi"], 1e-4)?;
    common::assert_max_diff(&lime_w, &w["lime_w_ref"], 1e-5)?;

    // 3. KernelSHAP combinatorial weights
    let shap_w = topic::shapley_kernel_weights(&w["shap_sizes"], 4)?;
    common::assert_max_diff(&shap_w, &w["shap_weights_ref"], 1e-5)?;

    // 4. RISE feature importance
    let rise_imp = topic::rise_feature_importance(&w["rise_masks"], &w["rise_outputs"])?;
    common::assert_max_diff(&rise_imp, &w["rise_importance_ref"], 1e-5)?;

    // 5. ROAR faithfulness degradation
    let roar_deg = topic::faithfulness_degradation(&w["roar_orig"], &w["roar_degraded"])?;
    common::assert_max_diff(&roar_deg, &w["roar_degradation_ref"], 1e-5)?;

    Ok(())
}
