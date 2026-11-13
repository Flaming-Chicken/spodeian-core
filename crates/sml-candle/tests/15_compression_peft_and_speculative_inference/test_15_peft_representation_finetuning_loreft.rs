use candle_core::{Device, Result, safetensors::load};

use ai_candle::c15_compression_peft_and_speculative_inference::t15_peft_representation_finetuning_loreft as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_15_peft_representation_finetuning_loreft_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/05_peft_representation_finetuning_loreft.safetensors",
        ),
        &device,
    )?;

    // Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 1. Delta H
    let delta_h =
        topic::loreft_subspace_delta(&w["h"], &w["r_matrix"], &w["w_matrix"], &w["b_vec"])?;
    common::assert_max_diff(&delta_h, &w["delta_h"], 1e-5)?;

    // 2. LoReFT Intervene
    let loreft_out = topic::loreft_intervene(
        &w["h"],
        &w["r_matrix"],
        &w["w_matrix"],
        &w["b_vec"],
        &w["position_mask"],
    )?;
    common::assert_max_diff(&loreft_out, &w["loreft_out"], 1e-5)?;

    Ok(())
}
