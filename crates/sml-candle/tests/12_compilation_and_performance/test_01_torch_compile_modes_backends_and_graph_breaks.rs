use candle_core::{safetensors::load, Device, Result};

use ai_candle::c12_compilation_and_performance::t01_torch_compile_modes_backends_and_graph_breaks as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_01_torch_compile_modes_backends_and_graph_breaks_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("12_compilation_and_performance/01_torch_compile_modes_backends_and_graph_breaks.safetensors"),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Fused SwiGLU
    let swiglu = topic::fused_swiglu(&w["swiglu_x"], &w["swiglu_gate"])?;
    common::assert_max_diff(&swiglu, &w["swiglu_ref"], 1e-5)?;

    // 3. Fused Add + LayerNorm
    let add_ln = topic::fused_add_layernorm(
        &w["ln_x"],
        &w["ln_residual"],
        &w["ln_gamma"],
        &w["ln_beta"],
        1e-5,
    )?;
    common::assert_max_diff(&add_ln, &w["add_ln_ref"], 1e-5)?;

    // 4. Branchless conditional selection
    let cond = topic::tensor_cond_select(&w["cond_x"], 0.0, 2.0, 0.5)?;
    common::assert_max_diff(&cond, &w["cond_ref"], 1e-5)?;

    Ok(())
}
