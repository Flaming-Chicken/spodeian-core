use candle_core::{safetensors::load, Device, Result};

use ai_candle::c12_compilation_and_performance::t02_dynamic_shapes_mark_dynamic_and_recompilation_guards as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_02_dynamic_shapes_mark_dynamic_and_recompilation_guards_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("12_compilation_and_performance/02_dynamic_shapes_mark_dynamic_and_recompilation_guards.safetensors"),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Dynamic masked pool
    let mp = topic::dynamic_masked_pool(&w["pool_x"], &w["pool_mask"])?;
    common::assert_max_diff(&mp, &w["masked_pool_ref"], 1e-5)?;

    // 3. Bucket allocation
    let alloc = topic::bucket_allocation(&w["seq_lens"], &w["buckets"])?;
    common::assert_max_diff(&alloc, &w["allocated_buckets_ref"], 1e-5)?;

    // 4. Dynamic RMSNorm
    let rms = topic::dynamic_rms_norm(&w["rms_x"], &w["rms_w"], 1e-5)?;
    common::assert_max_diff(&rms, &w["dynamic_rms_ref"], 1e-5)?;

    Ok(())
}
