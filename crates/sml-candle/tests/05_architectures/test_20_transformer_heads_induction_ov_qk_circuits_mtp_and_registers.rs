use candle_core::{Device, Result, safetensors::load};

use ai_candle::c05_architectures::t20_transformer_heads_induction_ov_qk_circuits_mtp_and_registers as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_20_transformer_heads_induction_ov_qk_circuits_mtp_and_registers_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "05_architectures/04_transformers_and_heads/02_transformer_heads_induction_ov_qk_circuits_mtp_and_registers.safetensors",
        ),
        &device,
    )?;

    // 1. Linear anchor parity
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. QK and OV circuits parity
    let scale = w["scale"].to_vec1::<f32>()?[0] as f64;
    let (out, attn) = topic::ov_qk_circuits(
        &w["x_q"], &w["x_k"], &w["x_v"], &w["w_q"], &w["w_k"], &w["w_v"], &w["w_o"], scale,
    )?;
    common::assert_max_diff(&out, &w["circuit_out"], 1e-5)?;
    common::assert_max_diff(&attn, &w["circuit_attn"], 1e-5)?;

    // 3. Registers and MTP parity
    let (x_aug, mtp_logits) =
        topic::register_prepend_and_mtp(&w["x_q"], &w["registers"], &w["w_mtp"])?;
    common::assert_max_diff(&x_aug, &w["x_aug"], 1e-5)?;
    common::assert_max_diff(&mtp_logits, &w["mtp_logits"], 1e-5)?;

    Ok(())
}
