use candle_core::{Device, Result, safetensors::load};

use ai_candle::c15_compression_peft_and_speculative_inference::t10_model_merging_slerp_task_arithmetic_ties_dare_and_evolutionary as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_10_model_merging_slerp_task_arithmetic_ties_dare_and_evolutionary_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging/05_model_merging_slerp_task_arithmetic_ties_dare_and_evolutionary.safetensors",
        ),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. SLERP
    let s_out = topic::slerp_merge(&w["v0"], &w["v1"], 0.3)?;
    common::assert_max_diff(&s_out, &w["slerp_out"], 1e-5)?;

    // 3. Task Arithmetic
    let a_out = topic::task_arithmetic_merge(&w["base_w"], &w["deltas"], &w["task_weights"])?;
    common::assert_max_diff(&a_out, &w["arithmetic_out"], 1e-5)?;

    // 4. TIES Merge
    let t_out = topic::ties_merge(&w["base_w"], &w["deltas"], 0.5, 0.6)?;
    common::assert_max_diff(&t_out, &w["ties_out"], 1e-5)?;

    // 5. DARE
    let d0 = w["deltas"].get(0)?;
    let d_out = topic::dare_rescale(&d0, &w["dare_mask"], 0.5)?;
    common::assert_max_diff(&d_out, &w["dare_out"], 1e-5)?;

    Ok(())
}
