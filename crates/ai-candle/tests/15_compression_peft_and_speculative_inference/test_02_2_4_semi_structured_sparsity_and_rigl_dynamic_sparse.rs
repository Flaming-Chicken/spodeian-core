use candle_core::{safetensors::load, Device, Result};

use ai_candle::c15_compression_peft_and_speculative_inference::t02_2_4_semi_structured_sparsity_and_rigl_dynamic_sparse as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_02_2_4_semi_structured_sparsity_and_rigl_dynamic_sparse_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/02_2_4_semi_structured_sparsity_and_rigl_dynamic_sparse.safetensors"),
        &device,
    )?;

    // 1. Linear anchor parity
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. 2:4 Semi-structured pruning
    let (w_24, mask_24) = topic::prune_2_to_4(&w["w"])?;
    common::assert_max_diff(&w_24, &w["w_2_to_4"], 1e-5)?;
    common::assert_max_diff(&mask_24, &w["mask_2_to_4"], 1e-5)?;

    // 3. 2:4 Storage compression
    let (packed_vals, packed_idxs) = topic::compress_2_to_4(&w["w"], &mask_24)?;
    common::assert_max_diff(&packed_vals, &w["packed_values"], 1e-5)?;
    common::assert_max_diff(&packed_idxs, &w["packed_indices"], 1e-5)?;

    // 4. RigL dynamic sparsity update
    let rigl_s1 = topic::rigl_update_mask(&w["w"], &w["grad"], &w["initial_mask"], 0.25)?;
    common::assert_max_diff(&rigl_s1, &w["rigl_mask_step1"], 1e-5)?;

    Ok(())
}
