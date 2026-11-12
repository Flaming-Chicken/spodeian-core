use candle_core::{safetensors::load, Device, Result};

use ai_candle::c12_compilation_and_performance::t04_torch_export_aot_autograd_and_ir_inspection as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_04_torch_export_aot_autograd_and_ir_inspection_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("12_compilation_and_performance/04_torch_export_aot_autograd_and_ir_inspection.safetensors"),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Core ATen Softmax
    let sm = topic::aten_core_softmax(&w["sm_x"])?;
    common::assert_max_diff(&sm, &w["sm_ref"], 1e-5)?;

    // 3. Core ATen LayerNorm
    let ln = topic::aten_core_layernorm(&w["ln_x"], &w["ln_gamma"], &w["ln_beta"], 1e-5)?;
    common::assert_max_diff(&ln, &w["ln_ref"], 1e-5)?;

    // 4. AOT Forward-Backward VJP
    let (fwd_a, grad_w, grad_b, grad_x) = topic::aot_forward_backward_affine_gelu(
        &w["mlp_x"],
        &w["mlp_w"],
        &w["mlp_b"],
        &w["mlp_grad_out"],
    )?;
    common::assert_max_diff(&fwd_a, &w["fwd_a_ref"], 1e-5)?;
    common::assert_max_diff(&grad_w, &w["grad_w_ref"], 1e-5)?;
    common::assert_max_diff(&grad_b, &w["grad_b_ref"], 1e-5)?;
    common::assert_max_diff(&grad_x, &w["grad_x_ref"], 1e-5)?;

    Ok(())
}
