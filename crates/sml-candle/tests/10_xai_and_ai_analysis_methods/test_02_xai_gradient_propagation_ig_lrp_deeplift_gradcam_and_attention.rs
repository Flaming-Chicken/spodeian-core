use candle_core::{safetensors::load, Device, Result};

use ai_candle::c10_xai_and_ai_analysis_methods::t02_xai_gradient_propagation_ig_lrp_deeplift_gradcam_and_attention as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_02_xai_gradient_propagation_ig_lrp_deeplift_gradcam_and_attention_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("10_xai_and_ai_analysis_methods/02_xai_gradient_propagation_ig_lrp_deeplift_gradcam_and_attention.safetensors"),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Integrated Gradients
    let ig = topic::integrated_gradients(&w["ig_x"], &w["ig_baseline"], &w["ig_grads"])?;
    common::assert_max_diff(&ig, &w["ig_ref"], 1e-5)?;

    // 3. LRP epsilon-rule
    let lrp = topic::lrp_epsilon_rule(&w["lrp_acts"], &w["lrp_weights"], &w["lrp_rel_next"], 1e-4)?;
    common::assert_max_diff(&lrp, &w["lrp_ref"], 1e-5)?;

    // 4. Grad-CAM 2D
    let cam = topic::grad_cam_2d(&w["cam_acts"], &w["cam_grads"])?;
    common::assert_max_diff(&cam, &w["cam_ref"], 1e-5)?;

    // 5. Adebayo randomization check
    let sim = topic::cosine_similarity(&w["adebayo_orig"], &w["adebayo_rand"], 1e-8)?;
    common::assert_max_diff(&sim, &w["adebayo_sim_ref"], 1e-5)?;

    Ok(())
}
