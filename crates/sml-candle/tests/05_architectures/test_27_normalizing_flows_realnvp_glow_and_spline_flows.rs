use candle_core::{Device, Result, safetensors::load};

use ai_candle::c05_architectures::t27_normalizing_flows_realnvp_glow_and_spline_flows as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_27_normalizing_flows_realnvp_glow_and_spline_flows_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "05_architectures/05_generative_models/04_normalizing_flows_realnvp_glow_and_spline_flows.safetensors",
        ),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. RealNVP forward
    let (y_a, y_b, log_det) = topic::realnvp_affine_coupling_forward(
        &w["x_a"], &w["x_b"], &w["s_w"], &w["s_b"], &w["t_w"], &w["t_b"],
    )?;
    common::assert_max_diff(&y_a, &w["y_a"], 1e-5)?;
    common::assert_max_diff(&y_b, &w["y_b"], 1e-5)?;
    common::assert_max_diff(&log_det, &w["log_det"], 1e-5)?;

    // 3. RealNVP inverse
    let (x_a_rec, x_b_rec, _) = topic::realnvp_affine_coupling_inverse(
        &y_a, &y_b, &w["s_w"], &w["s_b"], &w["t_w"], &w["t_b"],
    )?;
    common::assert_max_diff(&x_a_rec, &w["x_a_rec"], 1e-5)?;
    common::assert_max_diff(&x_b_rec, &w["x_b_rec"], 1e-5)?;

    // 4. Glow forward
    let y_glow = topic::glow_forward(&w["x_glow"], &w["w_glow"])?;
    common::assert_max_diff(&y_glow, &w["y_glow"], 1e-5)
}
