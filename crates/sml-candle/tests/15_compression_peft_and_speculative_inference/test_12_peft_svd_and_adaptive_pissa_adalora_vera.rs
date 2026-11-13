use candle_core::{Device, Result, safetensors::load};

use sml_candle::c15_compression_peft_and_speculative_inference::t12_peft_svd_and_adaptive_pissa_adalora_vera as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_12_peft_svd_and_adaptive_pissa_adalora_vera_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft/02_peft_svd_and_adaptive_pissa_adalora_vera.safetensors",
        ),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. PiSSA
    let p_out = topic::pissa_forward(&w["x"], &w["w_res"], &w["b_pissa"], &w["a_pissa"])?;
    common::assert_max_diff(&p_out, &w["pissa_out"], 1e-5)?;

    // 3. AdaLoRA
    let (a_out, d_w) = topic::adalora_forward(
        &w["x"],
        &w["w0"],
        &w["p"],
        &w["lambda_diag"],
        &w["q"],
        &w["ada_mask"],
    )?;
    common::assert_max_diff(&a_out, &w["ada_out"], 1e-5)?;
    common::assert_max_diff(&d_w, &w["delta_w"], 1e-5)?;

    // 4. VeRA
    let v_out = topic::vera_forward(
        &w["x"],
        &w["w0"],
        &w["a_shared"],
        &w["b_shared"],
        &w["d_vec"],
        &w["b_vec"],
    )?;
    common::assert_max_diff(&v_out, &w["vera_out"], 1e-5)?;

    Ok(())
}
