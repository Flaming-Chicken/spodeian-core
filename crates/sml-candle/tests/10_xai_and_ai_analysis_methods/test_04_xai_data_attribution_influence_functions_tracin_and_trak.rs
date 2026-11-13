use candle_core::{Device, Result, safetensors::load};

use ai_candle::c10_xai_and_ai_analysis_methods::t04_xai_data_attribution_influence_functions_tracin_and_trak as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_04_xai_data_attribution_influence_functions_tracin_and_trak_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "10_xai_and_ai_analysis_methods/04_xai_data_attribution_influence_functions_tracin_and_trak.safetensors",
        ),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Influence functions
    let (ihvp, inf_score) = topic::influence_score(
        &w["inf_hessian"],
        &w["inf_grad_test"],
        &w["inf_grad_train"],
        1e-3,
    )?;
    common::assert_max_diff(&ihvp, &w["inf_ihvp_ref"], 1e-5)?;
    common::assert_max_diff(&inf_score, &w["inf_score_ref"], 1e-5)?;

    // 3. TracIn
    let tracin = topic::tracin_trajectory_attribution(
        &w["tracin_train_g"],
        &w["tracin_test_g"],
        &w["tracin_lrs"],
    )?;
    common::assert_max_diff(&tracin, &w["tracin_score_ref"], 1e-5)?;

    // 4. TRAK
    let trak =
        topic::trak_projected_attribution(&w["trak_train_g"], &w["trak_test_g"], &w["trak_proj"])?;
    common::assert_max_diff(&trak, &w["trak_scores_ref"], 1e-5)?;

    Ok(())
}
