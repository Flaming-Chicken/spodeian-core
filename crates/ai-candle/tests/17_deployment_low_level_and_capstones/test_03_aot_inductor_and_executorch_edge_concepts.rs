use candle_core::{safetensors::load, Device, Result};

use ai_candle::c17_deployment_low_level_and_capstones::t03_aot_inductor_and_executorch_edge_concepts as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_03_aot_inductor_and_executorch_edge_concepts_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("17_deployment_low_level_and_capstones/03_aot_inductor_and_executorch_edge_concepts.safetensors"),
        &device,
    )?;

    // Anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // Edge forward: Linear -> LayerNorm -> SiLU
    let (y_out, z, hat_z) =
        topic::aot_edge_forward(&w["x"], &w["w"], &w["b"], &w["gamma"], &w["beta"])?;
    common::assert_max_diff(&y_out, &w["y_out"], 1e-5)?;
    common::assert_max_diff(&z, &w["linear_z"], 1e-5)?;
    common::assert_max_diff(&hat_z, &w["norm_z"], 1e-5)?;

    // Arena offsets plan
    let offsets = topic::plan_static_memory_arena(&[64, 64, 64, 64], &[(1, 3)], &device)?;
    let offsets_f32 = offsets.to_dtype(candle_core::DType::F32)?;
    let expected_offsets_f32 = w["arena_offsets"].to_dtype(candle_core::DType::F32)?;
    common::assert_max_diff(&offsets_f32, &expected_offsets_f32, 1e-5)?;

    Ok(())
}
