use candle_core::{Device, Result, safetensors::load};

use sml_candle::c05_architectures::t29_diffusion_conditioning_controlnet_and_ip_adapter as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_29_diffusion_conditioning_controlnet_and_ip_adapter_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "05_architectures/05_generative_models/06_diffusion_conditioning_controlnet_and_ip_adapter.safetensors",
        ),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. ControlNet zero-conv
    let (y_ctrl, _) = topic::controlnet_residual_block(
        &w["x"],
        &w["control_signal"],
        &w["backbone_w"],
        &w["control_w"],
        &w["z_in_w"],
        &w["z_in_b"],
        &w["z_out_w"],
        &w["z_out_b"],
    )?;
    common::assert_max_diff(&y_ctrl, &w["y_ctrl"], 1e-5)?;

    // 3. ControlNet adapted
    let (y_adapted, _) = topic::controlnet_residual_block(
        &w["x"],
        &w["control_signal"],
        &w["backbone_w"],
        &w["control_w"],
        &w["z_in_w_adapt"],
        &w["z_in_b"],
        &w["z_out_w_adapt"],
        &w["z_out_b"],
    )?;
    common::assert_max_diff(&y_adapted, &w["y_adapted"], 1e-5)?;

    // 4. IP-Adapter decoupled cross-attention
    let scale = w["scale"].to_vec1::<f32>()?[0] as f64;
    let lambda_img = w["lambda_img"].to_vec1::<f32>()?[0] as f64;
    let ip_out = topic::ip_adapter_decoupled_cross_attention(
        &w["q"],
        &w["k_text"],
        &w["v_text"],
        &w["k_img"],
        &w["v_img"],
        scale,
        lambda_img,
    )?;
    common::assert_max_diff(&ip_out, &w["ip_out"], 1e-5)
}
