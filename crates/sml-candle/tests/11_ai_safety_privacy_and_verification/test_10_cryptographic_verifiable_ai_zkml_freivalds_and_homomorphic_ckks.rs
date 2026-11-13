use candle_core::{Device, Result, safetensors::load};

use sml_candle::c11_ai_safety_privacy_and_verification::t10_cryptographic_verifiable_ai_zkml_freivalds_and_homomorphic_ckks as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_10_cryptographic_verifiable_ai_zkml_freivalds_and_homomorphic_ckks_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "11_ai_safety_privacy_and_verification/10_cryptographic_verifiable_ai_zkml_freivalds_and_homomorphic_ckks.safetensors",
        ),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Freivalds valid residual
    let v_val =
        topic::freivalds_verify_residual(&w["mat_a"], &w["mat_b"], &w["mat_c"], &w["rand_r"])?;
    common::assert_max_diff(&v_val, &w["freivalds_valid_ref"], 1e-5)?;

    // 3. Freivalds tampered residual
    let v_tamp = topic::freivalds_verify_residual(
        &w["mat_a"],
        &w["mat_b"],
        &w["mat_c_tampered"],
        &w["rand_r"],
    )?;
    common::assert_max_diff(&v_tamp, &w["freivalds_tampered_ref"], 1e-5)?;

    // 4. CKKS degree-2
    let p2 = topic::ckks_degree2_activation(&w["x_poly"], 0.125, 0.5, 0.25)?;
    common::assert_max_diff(&p2, &w["poly2_out_ref"], 1e-5)?;

    // 5. CKKS degree-4
    let p4 = topic::ckks_degree4_activation(&w["x_poly"], -0.005, 0.2, 0.15)?;
    common::assert_max_diff(&p4, &w["poly4_out_ref"], 1e-5)?;

    // 6. ZKML fixed-point linear
    let zk_out = topic::zkml_fixed_point_linear(&w["zk_x"], &w["zk_w"], &w["zk_b"], 16.0)?;
    common::assert_max_diff(&zk_out, &w["zk_out_ref"], 1e-5)?;

    Ok(())
}
