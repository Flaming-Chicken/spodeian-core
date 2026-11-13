use candle_core::{Device, Result, safetensors::load};

use sml_candle::c15_compression_peft_and_speculative_inference::t11_peft_lora_rslora_loraplus_and_dora as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_11_peft_lora_rslora_loraplus_and_dora_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/01_peft_lora_rslora_loraplus_and_dora.safetensors",
        ),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. LoRA
    let l_out = topic::lora_forward(&w["x"], &w["w0"], &w["a"], &w["b"], 4.0, 2)?;
    common::assert_max_diff(&l_out, &w["lora_out"], 1e-5)?;

    // 3. rsLoRA
    let rs_out = topic::rslora_forward(&w["x"], &w["w0"], &w["a"], &w["b"], 4.0, 2)?;
    common::assert_max_diff(&rs_out, &w["rslora_out"], 1e-5)?;

    // 4. DoRA
    let d_out = topic::dora_forward(&w["x"], &w["w0"], &w["a"], &w["b"], &w["m"], 4.0, 2)?;
    common::assert_max_diff(&d_out, &w["dora_out"], 1e-5)?;

    Ok(())
}
