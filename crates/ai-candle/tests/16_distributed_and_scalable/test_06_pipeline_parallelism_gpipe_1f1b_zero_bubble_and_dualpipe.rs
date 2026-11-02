use candle_core::{safetensors::load, Device, Result};

use ai_candle::c16_distributed_and_scalable::t06_pipeline_parallelism_gpipe_1f1b_zero_bubble_and_dualpipe as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_06_pipeline_parallelism_gpipe_1f1b_zero_bubble_and_dualpipe_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("16_distributed_and_scalable/06_pipeline_parallelism_gpipe_1f1b_zero_bubble_and_dualpipe.safetensors"),
        &device,
    )?;

    // Anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // Bubble fraction check
    let bubble_f = topic::compute_pipeline_bubble_fraction(4, 4);
    assert!((bubble_f - 3.0 / 7.0).abs() < 1e-6);

    // Forward
    let (h_mb, y_mb) =
        topic::pipeline_forward_step(&w["x_mb"], &w["w0"], &w["b0"], &w["w1"], &w["b1"])?;
    common::assert_max_diff(&h_mb, &w["stage0_fwd_outs"], 1e-5)?;
    common::assert_max_diff(&y_mb, &w["stage1_fwd_outs"], 1e-5)?;

    // Backward
    let (grad_w0, grad_w1, grad_x) = topic::dualpipe_backward_step(
        &w["x_mb"],
        &w["stage0_fwd_outs"],
        &w["grad_y_mb"],
        &w["w0"],
        &w["w1"],
    )?;
    common::assert_max_diff(&grad_w0, &w["grad_w0"], 1e-5)?;
    common::assert_max_diff(&grad_w1, &w["grad_w1"], 1e-5)?;
    common::assert_max_diff(&grad_x, &w["grad_x_outs"], 1e-5)?;

    Ok(())
}
