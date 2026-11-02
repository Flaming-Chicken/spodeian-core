use candle_core::{safetensors::load, Device, Result};

use ai_candle::c17_deployment_low_level_and_capstones::t08_triton_and_tiled_kernel_algorithms_softmax_rmsnorm_liger_ce as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_08_triton_and_tiled_kernel_algorithms_softmax_rmsnorm_liger_ce_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "17_deployment_low_level_and_capstones/08_triton_and_tiled_kernel_algorithms_softmax_rmsnorm_liger_ce.safetensors",
        ),
        &device,
    )?;

    let y_lin = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y_lin, &w["test_output"], 1e-5)?;

    let normed = topic::fused_rmsnorm(&w["h"], &w["gamma"], 1e-5)?;
    common::assert_max_diff(&normed, &w["normed_h"], 1e-5)?;

    let swiglu = topic::fused_swiglu(&w["h"], &w["w_gate"], &w["w_up"])?;
    common::assert_max_diff(&swiglu, &w["swiglu_out"], 1e-5)?;

    let (loss, dh) = topic::tiled_linear_cross_entropy(&w["h"], &w["w_vocab"], &w["labels"])?;
    common::assert_max_diff(&loss, &w["loss"], 1e-5)?;
    common::assert_max_diff(&dh, &w["dh"], 1e-5)?;

    Ok(())
}
