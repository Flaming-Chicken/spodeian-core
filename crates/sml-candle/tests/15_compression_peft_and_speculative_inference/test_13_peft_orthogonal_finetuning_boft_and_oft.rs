use candle_core::{safetensors::load, Device, Result};

use ai_candle::c15_compression_peft_and_speculative_inference::t13_peft_orthogonal_finetuning_boft_and_oft as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_13_peft_orthogonal_finetuning_boft_and_oft_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(common::get_weights_path("15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/03_peft_orthogonal_finetuning_boft_and_oft.safetensors"), &device)?;

    // Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 1. Cayley orthogonal
    let r_cayley = topic::cayley_orthogonal(&w["q_skew"])?;
    common::assert_max_diff(&r_cayley, &w["r_cayley"], 1e-5)?;

    // 2. Block OFT
    let (oft_out, r_full) = topic::block_oft_forward(&w["x"], &w["w0"], &w["q_blocks1"])?;
    common::assert_max_diff(&oft_out, &w["oft_out"], 1e-5)?;
    common::assert_max_diff(&r_full, &w["r_full"], 1e-5)?;

    // 3. Butterfly BOFT
    let (boft_out, r_total) =
        topic::butterfly_boft_forward(&w["x"], &w["w0"], &w["q_blocks1"], &w["q_blocks2"])?;
    common::assert_max_diff(&boft_out, &w["boft_out"], 1e-5)?;
    common::assert_max_diff(&r_total, &w["r_total"], 1e-5)?;

    Ok(())
}
