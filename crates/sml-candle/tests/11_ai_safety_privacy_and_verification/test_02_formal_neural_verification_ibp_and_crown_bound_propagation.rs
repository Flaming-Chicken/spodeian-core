use candle_core::{safetensors::load, Device, Result};

use ai_candle::c11_ai_safety_privacy_and_verification::t02_formal_neural_verification_ibp_and_crown_bound_propagation as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_02_formal_neural_verification_ibp_and_crown_bound_propagation_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("11_ai_safety_privacy_and_verification/02_formal_neural_verification_ibp_and_crown_bound_propagation.safetensors"),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. IBP Linear
    let (z1_l, z1_u) = topic::ibp_linear(&w["x_l"], &w["x_u"], &w["w1"], &w["b1"])?;
    common::assert_max_diff(&z1_l, &w["z1_l_ref"], 1e-5)?;
    common::assert_max_diff(&z1_u, &w["z1_u_ref"], 1e-5)?;

    // 3. IBP ReLU
    let (a1_l, a1_u) = topic::ibp_relu(&z1_l, &z1_u)?;
    common::assert_max_diff(&a1_l, &w["a1_l_ref"], 1e-5)?;
    common::assert_max_diff(&a1_u, &w["a1_u_ref"], 1e-5)?;

    // 4. CROWN ReLU relaxation
    let (s_l, b_l, s_u, b_u) = topic::crown_relu_relaxation(&z1_l, &z1_u)?;
    common::assert_max_diff(&s_l, &w["s_L_ref"], 1e-5)?;
    common::assert_max_diff(&b_l, &w["b_L_ref"], 1e-5)?;
    common::assert_max_diff(&s_u, &w["s_U_ref"], 1e-5)?;
    common::assert_max_diff(&b_u, &w["b_U_ref"], 1e-5)?;

    // 5. CROWN 2-layer verified bounds
    let (crown_l, crown_u) = topic::crown_2layer_bounds(
        &w["x_l"], &w["x_u"], &w["w1"], &w["b1"], &w["w2"], &w["b2"], &w["c"],
    )?;
    common::assert_max_diff(&crown_l, &w["crown_lower_ref"], 1e-5)?;
    common::assert_max_diff(&crown_u, &w["crown_upper_ref"], 1e-5)?;

    Ok(())
}
