use candle_core::{safetensors::load, Device, Result};

use ai_candle::c14_quantisation::t03_quantile_and_information_theoretic_nf4_sf4_fp4_lloyd_max_and_dynamic_tree as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_03_quantile_and_information_theoretic_nf4_sf4_fp4_lloyd_max_and_dynamic_tree_parity(
) -> Result<()> {
    let device = Device::Cpu;
    let w = load(common::get_weights_path("14_quantisation/01_foundations_formats_and_observers/03_quantile_and_information_theoretic_nf4_sf4_fp4_lloyd_max_and_dynamic_tree.safetensors"), &device)?;

    // Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // NF4 Quantization
    let (nf4_dequant, nf4_scales) = topic::nf4_quantize_dequantize(&w["raw_gaussian"], 64)?;
    common::assert_max_diff(&nf4_dequant, &w["nf4_dequant_ref"], 1e-5)?;
    common::assert_max_diff(&nf4_scales, &w["nf4_scales_ref"], 1e-5)?;

    // SF4 Quantization
    let (sf4_dequant, sf4_scales) = topic::sf4_quantize_dequantize(&w["raw_heavy"], 64)?;
    common::assert_max_diff(&sf4_dequant, &w["sf4_dequant_ref"], 1e-5)?;
    common::assert_max_diff(&sf4_scales, &w["sf4_scales_ref"], 1e-5)?;

    // Lloyd-Max Quantization
    let (lloyd_q, lloyd_centroids) = topic::lloyd_max_scalar_quantize(&w["lloyd_input"], 8, 15)?;
    common::assert_max_diff(&lloyd_q, &w["lloyd_q_ref"], 1e-5)?;
    common::assert_max_diff(&lloyd_centroids, &w["lloyd_centroids_ref"], 1e-5)?;

    // Dynamic Tree Quantization
    let tree_q = topic::dynamic_tree_scalar_quantize(&w["tree_input"], 3)?;
    common::assert_max_diff(&tree_q, &w["tree_q_ref"], 1e-5)?;

    Ok(())
}
