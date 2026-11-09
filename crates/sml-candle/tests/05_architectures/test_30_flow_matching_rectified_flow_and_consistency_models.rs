use candle_core::{safetensors::load, Device, Result};

use ai_candle::c05_architectures::t30_flow_matching_rectified_flow_and_consistency_models as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_30_flow_matching_rectified_flow_and_consistency_models_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("05_architectures/05_generative_models/07_flow_matching_rectified_flow_and_consistency_models.safetensors"),
        &device,
    )?;

    // Linear anchor check
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // Flow matching linear interpolation
    let (x_t, v_target) = topic::conditional_flow_matching(&w["x_0"], &w["x_1"], &w["t"])?;
    common::assert_max_diff(&x_t, &w["x_t"], 1e-5)?;
    common::assert_max_diff(&v_target, &w["v_target"], 1e-5)?;

    // Velocity network
    let v_pred = topic::velocity_network(
        &x_t,
        &w["t"],
        &w["w_x"],
        &w["b_x"],
        &w["w_t"],
        &w["b_t"],
        &w["w_out"],
        &w["b_out"],
    )?;
    common::assert_max_diff(&v_pred, &w["v_pred"], 1e-5)?;

    // CFM Loss
    let diff = (v_pred.clone() - &w["v_target"])?;
    let cfm_loss = diff.sqr()?.mean_all()?.reshape((1,))?;
    common::assert_max_diff(&cfm_loss, &w["cfm_loss"], 1e-5)?;

    // Rectified flow step
    let x_next = topic::rectified_flow_step(&x_t, &v_pred, 0.1)?;
    common::assert_max_diff(&x_next, &w["x_next"], 1e-5)?;

    // Consistency coefficients
    let (c_skip, c_out) = topic::consistency_coefficients(&w["t"], 0.002, 0.5)?;
    common::assert_max_diff(&c_skip, &w["c_skip"], 1e-5)?;
    common::assert_max_diff(&c_out, &w["c_out"], 1e-5)?;

    // Consistency forward
    let cons_out = topic::consistency_forward(&x_t, &w["t"], &v_pred, 0.002, 0.5)?;
    common::assert_max_diff(&cons_out, &w["cons_out"], 1e-5)?;

    Ok(())
}
