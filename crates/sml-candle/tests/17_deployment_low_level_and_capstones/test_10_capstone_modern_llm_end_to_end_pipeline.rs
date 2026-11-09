use candle_core::{safetensors::load, Device, Result};

use ai_candle::c17_deployment_low_level_and_capstones::t10_capstone_modern_llm_end_to_end_pipeline as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_10_capstone_modern_llm_end_to_end_pipeline_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "17_deployment_low_level_and_capstones/10_capstone_modern_llm_end_to_end_pipeline.safetensors",
        ),
        &device,
    )?;

    let y_lin = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y_lin, &w["test_output"], 1e-5)?;

    let (logits, h_out, attn, ffn) = topic::modern_llm_pipeline_forward(
        &w["x"],
        &w["gamma1"],
        &w["wq"],
        &w["wk"],
        &w["wv"],
        &w["wo"],
        &w["gamma2"],
        &w["w_gate"],
        &w["w_up"],
        &w["w_down"],
        &w["gamma_final"],
        &w["w_head"],
    )?;

    common::assert_max_diff(&logits, &w["logits"], 1e-5)?;
    common::assert_max_diff(&h_out, &w["h_out"], 1e-5)?;
    common::assert_max_diff(&attn, &w["attn"], 1e-5)?;
    common::assert_max_diff(&ffn, &w["ffn"], 1e-5)?;

    Ok(())
}
