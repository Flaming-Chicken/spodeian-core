use candle_core::{Device, Result, safetensors::load};

use ai_candle::c16_distributed_and_scalable::t07_expert_parallelism_and_3d_4d_parallel_composition as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_07_expert_parallelism_and_3d_4d_parallel_composition_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "16_distributed_and_scalable/07_expert_parallelism_and_3d_4d_parallel_composition.safetensors",
        ),
        &device,
    )?;

    // Anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // Mesh coordinates
    let coords = topic::compute_3d_mesh_coordinates(2, 2, 2, &device)?;
    let coords_f32 = coords.to_dtype(candle_core::DType::F32)?;
    let expected_coords_f32 = w["mesh_coords"].to_dtype(candle_core::DType::F32)?;
    common::assert_max_diff(&coords_f32, &expected_coords_f32, 1e-5)?;

    // Expert Parallel forward
    let (moe_out, top2_indices, top2_probs, dispatch_mask) =
        topic::expert_parallel_forward(&w["x"], &w["w_gate"], &w["w_experts"], 2)?;
    common::assert_max_diff(&moe_out, &w["moe_out"], 1e-5)?;
    common::assert_max_diff(&top2_probs, &w["top2_probs"], 1e-5)?;
    common::assert_max_diff(&dispatch_mask, &w["dispatch_mask"], 1e-5)?;

    let indices_f32 = top2_indices.to_dtype(candle_core::DType::F32)?;
    let expected_indices_f32 = w["top2_indices"].to_dtype(candle_core::DType::F32)?;
    common::assert_max_diff(&indices_f32, &expected_indices_f32, 1e-5)?;

    Ok(())
}
