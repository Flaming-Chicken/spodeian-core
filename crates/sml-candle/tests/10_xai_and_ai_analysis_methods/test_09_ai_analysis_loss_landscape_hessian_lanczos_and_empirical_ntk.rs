use candle_core::{safetensors::load, Device, Result};

use ai_candle::c10_xai_and_ai_analysis_methods::t09_ai_analysis_loss_landscape_hessian_lanczos_and_empirical_ntk as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_09_ai_analysis_loss_landscape_hessian_lanczos_and_empirical_ntk_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("10_xai_and_ai_analysis_methods/09_ai_analysis_loss_landscape_hessian_lanczos_and_empirical_ntk.safetensors"),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Filter normalize direction & loss slice
    let d1_norm = topic::filter_normalize_direction(&w["ls_theta"], &w["ls_d1_raw"])?;
    common::assert_max_diff(&d1_norm, &w["ls_d1_norm"], 1e-5)?;

    let d2_norm = topic::filter_normalize_direction(&w["ls_theta"], &w["ls_d2_raw"])?;
    common::assert_max_diff(&d2_norm, &w["ls_d2_norm"], 1e-5)?;

    let ls_slice = topic::loss_surface_coord(&w["ls_theta"], &d1_norm, &d2_norm, 0.5, -0.25)?;
    common::assert_max_diff(&ls_slice, &w["ls_slice_ref"], 1e-5)?;

    // 3. Lanczos tridiagonal
    let (alpha_diag, beta_diag) = topic::lanczos_tridiagonal(&w["h_mat"], &w["v0"], 3)?;
    common::assert_max_diff(&alpha_diag, &w["lanczos_alpha_ref"], 1e-5)?;
    common::assert_max_diff(&beta_diag, &w["lanczos_beta_ref"], 1e-5)?;

    // 4. Empirical NTK
    let ntk = topic::empirical_ntk(&w["ntk_j"])?;
    common::assert_max_diff(&ntk, &w["ntk_gram_ref"], 1e-5)?;

    Ok(())
}
