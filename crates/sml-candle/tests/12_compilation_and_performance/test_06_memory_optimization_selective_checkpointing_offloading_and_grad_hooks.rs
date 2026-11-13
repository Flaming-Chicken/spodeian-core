use candle_core::{Device, Result, safetensors::load};

use sml_candle::c12_compilation_and_performance::t06_memory_optimization_selective_checkpointing_offloading_and_grad_hooks as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_06_memory_optimization_selective_checkpointing_offloading_and_grad_hooks_parity()
-> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "12_compilation_and_performance/06_memory_optimization_selective_checkpointing_offloading_and_grad_hooks.safetensors",
        ),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Selective Checkpointing SiLU Gate
    let (chk_y, grad_w, grad_b, grad_x, grad_gate) = topic::selective_checkpoint_silu_gate(
        &w["chk_x"],
        &w["chk_w"],
        &w["chk_b"],
        &w["chk_gate"],
        &w["chk_grad_out"],
    )?;
    common::assert_max_diff(&chk_y, &w["chk_y_ref"], 1e-5)?;
    common::assert_max_diff(&grad_w, &w["chk_grad_w_ref"], 1e-5)?;
    common::assert_max_diff(&grad_b, &w["chk_grad_b_ref"], 1e-5)?;
    common::assert_max_diff(&grad_x, &w["chk_grad_x_ref"], 1e-5)?;
    common::assert_max_diff(&grad_gate, &w["chk_grad_gate_ref"], 1e-5)?;

    // 3. Post-Accumulate Grad Hook step
    let (w_new, m_new) = topic::post_accumulate_grad_hook_step(
        &w["chk_w"],
        &w["chk_grad_w_ref"],
        &w["hook_running_m"],
        0.9,
        0.01,
    )?;
    common::assert_max_diff(&w_new, &w["hook_w_new_ref"], 1e-5)?;
    common::assert_max_diff(&m_new, &w["hook_m_new_ref"], 1e-5)?;

    Ok(())
}
