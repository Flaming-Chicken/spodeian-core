use candle_core::{safetensors::load, Device, Result};

use ai_candle::c14_quantisation::t05_pt2e_export_quantization_workflow as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_05_pt2e_export_quantization_workflow_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(common::get_weights_path("14_quantisation/02_pt2e_torchao_and_bitpacking/01_pt2e_export_quantization_workflow.safetensors"), &device)?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Conv-BN fold
    let (wf, bf) = topic::conv2d_bn_fold(
        &w["w_conv"],
        &w["b_conv"],
        &w["bn_mean"],
        &w["bn_var"],
        &w["bn_gamma"],
        &w["bn_beta"],
        1e-5,
    )?;
    common::assert_max_diff(&wf, &w["w_folded"], 1e-5)?;
    common::assert_max_diff(&bf, &w["b_folded"], 1e-5)?;

    // 3. Linear-BN fold
    let (wlf, blf) = topic::linear_bn_fold(
        &w["w_lin"],
        &w["b_lin"],
        &w["lin_mean"],
        &w["lin_var"],
        &w["lin_gamma"],
        &w["lin_beta"],
        1e-5,
    )?;
    common::assert_max_diff(&wlf, &w["w_lin_folded"], 1e-5)?;
    common::assert_max_diff(&blf, &w["b_lin_folded"], 1e-5)?;

    // 4. QDQ Conv2d
    let s_x_conv = w["x_conv"]
        .flatten_all()?
        .to_vec1::<f32>()?
        .iter()
        .map(|v| v.abs())
        .fold(0.0f32, f32::max)
        / 127.0;
    let s_w_conv = w["w_folded"]
        .flatten_all()?
        .to_vec1::<f32>()?
        .iter()
        .map(|v| v.abs())
        .fold(0.0f32, f32::max)
        / 127.0;
    let out_qdq_conv = topic::pt2e_qdq_conv2d(
        &w["x_conv"],
        &w["w_folded"],
        &w["b_folded"],
        s_x_conv,
        s_w_conv,
    )?;
    common::assert_max_diff(&out_qdq_conv, &w["out_qdq_conv"], 1e-5)?;

    // 5. QDQ Linear
    let s_x_lin = w["x_lin"]
        .flatten_all()?
        .to_vec1::<f32>()?
        .iter()
        .map(|v| v.abs())
        .fold(0.0f32, f32::max)
        / 127.0;
    let s_w_lin = w["w_lin_folded"]
        .flatten_all()?
        .to_vec1::<f32>()?
        .iter()
        .map(|v| v.abs())
        .fold(0.0f32, f32::max)
        / 127.0;
    let out_qdq_lin = topic::pt2e_qdq_linear(
        &w["x_lin"],
        &w["w_lin_folded"],
        &w["b_lin_folded"],
        s_x_lin,
        s_w_lin,
    )?;
    common::assert_max_diff(&out_qdq_lin, &w["out_qdq_lin"], 1e-5)?;

    // 6. Unfused Conv-BN
    let out_unf = topic::unfused_conv_bn_quantize(
        &w["x_conv"],
        &w["w_conv"],
        &w["b_conv"],
        &w["bn_mean"],
        &w["bn_var"],
        &w["bn_gamma"],
        &w["bn_beta"],
        s_w_conv,
        1e-5,
    )?;
    common::assert_max_diff(&out_unf, &w["out_unfused"], 1e-5)?;

    Ok(())
}
