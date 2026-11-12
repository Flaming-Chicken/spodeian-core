use candle_core::{safetensors::load, Device, Result};

use ai_candle::c14_quantisation::t12_rotation_and_incoherence_ptq_quarot_spinquant_and_quip_sharp as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_12_rotation_and_incoherence_ptq_quarot_spinquant_and_quip_sharp_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(common::get_weights_path("14_quantisation/03_llm_ptq_algorithms/05_rotation_and_incoherence_ptq_quarot_spinquant_and_quip_sharp.safetensors"), &device)?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Hadamard matrix & Randomized Hadamard
    let h16 = topic::construct_hadamard_matrix(16, &device)?;
    common::assert_max_diff(&h16, &w["hadamard_16"], 1e-5)?;

    let q_had = topic::construct_randomized_hadamard(16, &w["signs"])?;
    common::assert_max_diff(&q_had, &w["q_hadamard"], 1e-5)?;

    // 3. Rotated activations and weights
    let x_rot = w["x_calib"].matmul(&w["q_hadamard"])?;
    let w_rot = w["q_hadamard"].t()?.matmul(&w["w_orig"])?;
    common::assert_max_diff(&x_rot, &w["x_rot"], 1e-5)?;
    common::assert_max_diff(&w_rot, &w["w_rot"], 1e-5)?;

    // 4. W4A4 outputs
    let out_unrot = topic::w4a4_gemm(&w["x_calib"], &w["w_orig"])?.broadcast_add(&w["b_orig"])?;
    let out_quarot = topic::w4a4_gemm(&x_rot, &w_rot)?.broadcast_add(&w["b_orig"])?;
    common::assert_max_diff(&out_unrot, &w["out_unrotated_w4a4"], 1e-5)?;
    common::assert_max_diff(&out_quarot, &w["out_quarot_w4a4"], 1e-5)?;

    Ok(())
}
