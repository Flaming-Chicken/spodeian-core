use candle_core::{Device, Result, safetensors::load};

use sml_candle::c10_xai_and_ai_analysis_methods::t07_ai_analysis_representation_similarity_cka_svcca_rsa_and_intrinsic_dim as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_07_ai_analysis_representation_similarity_cka_svcca_rsa_and_intrinsic_dim_parity()
-> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "10_xai_and_ai_analysis_methods/07_ai_analysis_representation_similarity_cka_svcca_rsa_and_intrinsic_dim.safetensors",
        ),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Linear CKA
    let cka = topic::linear_cka(&w["cka_x"], &w["cka_y"])?;
    common::assert_max_diff(&cka, &w["cka_score_ref"], 1e-5)?;

    // 3. RSA correlation
    let rsa = topic::rsa_correlation(&w["rdm_a"], &w["rdm_b"])?;
    common::assert_max_diff(&rsa, &w["rsa_corr_ref"], 1e-5)?;

    // 4. TwoNN intrinsic dimension
    let id = topic::twonn_intrinsic_dimension(&w["twonn_r1"], &w["twonn_r2"])?;
    common::assert_max_diff(&id, &w["twonn_id_ref"], 1e-5)?;

    Ok(())
}
