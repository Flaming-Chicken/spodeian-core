use candle_core::{Device, Result, safetensors::load};

use sml_candle::c13_precision_and_numerics::t05_block_floating_point_bfp_microscaling_mxfp_and_nvfp4 as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_05_block_floating_point_bfp_microscaling_mxfp_and_nvfp4_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "13_precision_and_numerics/05_block_floating_point_bfp_microscaling_mxfp_and_nvfp4.safetensors",
        ),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Block Floating Point
    let (bfp_dequant, bfp_scales) = topic::block_floating_point_quantize(&w["bfp_input"], 16, 8)?;
    common::assert_max_diff(&bfp_dequant, &w["bfp_dequant_ref"], 1e-5)?;
    common::assert_max_diff(&bfp_scales, &w["bfp_scales_ref"], 1e-5)?;

    // 3. OCP MXFP4
    let (mx_dequant, mx_scales) = topic::ocp_mxfp4_quantize(&w["mx_input"], 32)?;
    common::assert_max_diff(&mx_dequant, &w["mx_dequant_ref"], 1e-5)?;
    common::assert_max_diff(&mx_scales, &w["mx_scales_ref"], 1e-5)?;

    // 4. NVFP4 Two-Level
    let (nv_dequant, nv_glob, nv_loc) = topic::nvfp4_two_level_quantize(&w["nv_input"], 16)?;
    common::assert_max_diff(&nv_dequant, &w["nv_dequant_ref"], 1e-5)?;
    let ref_glob = w["nv_global_scale"].to_scalar::<f32>()? as f64;
    assert!(
        (nv_glob - ref_glob).abs() < 1e-5,
        "nv_glob mismatch: {} vs {}",
        nv_glob,
        ref_glob
    );
    common::assert_max_diff(&nv_loc, &w["nv_local_scales_ref"], 1e-5)?;

    Ok(())
}
