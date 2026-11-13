use candle_core::{Device, Result, safetensors::load};

use ai_candle::c13_precision_and_numerics::t04_fp8_e4m3_e5m2_fnuz_and_minifloats_fp6_fp4 as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_04_fp8_e4m3_e5m2_fnuz_and_minifloats_fp6_fp4_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "13_precision_and_numerics/04_fp8_e4m3_e5m2_fnuz_and_minifloats_fp6_fp4.safetensors",
        ),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. FP8 E4M3FN
    let e4m3 = topic::fp8_e4m3fn_quantize(&w["fp8_input"], 0.5)?;
    common::assert_max_diff(&e4m3, &w["e4m3_ref"], 1e-5)?;

    // 3. FP8 E5M2
    let e5m2 = topic::fp8_e5m2_quantize(&w["fp8_input"], 0.5)?;
    common::assert_max_diff(&e5m2, &w["e5m2_ref"], 1e-5)?;

    // 4. FP4 E2M1
    let fp4 = topic::fp4_e2m1_quantize(&w["fp4_input"], 0.25)?;
    common::assert_max_diff(&fp4, &w["fp4_ref"], 1e-5)?;

    // 5. Delayed scaled FP8 GEMM
    let (out, next_a, next_b) =
        topic::delayed_scaled_fp8_gemm(&w["gemm_a"], &w["gemm_b"], 0.01, 0.01)?;
    common::assert_max_diff(&out, &w["gemm_out_ref"], 1e-5)?;
    let ref_next_a = w["next_scale_a"].to_scalar::<f32>()? as f64;
    let ref_next_b = w["next_scale_b"].to_scalar::<f32>()? as f64;
    assert!(
        (next_a - ref_next_a).abs() < 1e-5,
        "next_a mismatch: {} vs {}",
        next_a,
        ref_next_a
    );
    assert!(
        (next_b - ref_next_b).abs() < 1e-5,
        "next_b mismatch: {} vs {}",
        next_b,
        ref_next_b
    );

    Ok(())
}
