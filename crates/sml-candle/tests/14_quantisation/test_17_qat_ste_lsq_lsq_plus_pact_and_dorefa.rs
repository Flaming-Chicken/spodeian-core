use candle_core::{safetensors::load, Device, Result};

use ai_candle::c14_quantisation::t17_qat_ste_lsq_lsq_plus_pact_and_dorefa as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_17_qat_ste_lsq_lsq_plus_pact_and_dorefa_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("14_quantisation/04_vector_lattice_qat_and_lowbit/03_qat_ste_lsq_lsq_plus_pact_and_dorefa.safetensors"),
        &device,
    )?;

    // 1. Linear anchor parity
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. STE parity
    let (w_ste_deq, ste_mask) = topic::ste_quantize(&w["w_orig"], &w["scale_w"], -8.0, 7.0)?;
    common::assert_max_diff(&w_ste_deq, &w["w_ste_deq"], 1e-5)?;
    common::assert_max_diff(&ste_mask, &w["ste_mask"], 1e-5)?;

    // 3. PACT parity
    let (x_pact_deq, pact_grad_mask) = topic::pact_activation(&w["x_calib"], &w["alpha_x"], 4)?;
    common::assert_max_diff(&x_pact_deq, &w["x_pact_deq"], 1e-5)?;
    common::assert_max_diff(&pact_grad_mask, &w["pact_grad_mask"], 1e-5)?;

    // 4. LSQ parity
    let (w_lsq_deq, lsq_grad_scale) = topic::lsq_quantize(&w["w_orig"], &w["scale_w"], -8.0, 7.0)?;
    common::assert_max_diff(&w_lsq_deq, &w["w_lsq_deq"], 1e-5)?;
    common::assert_max_diff(&lsq_grad_scale, &w["lsq_grad_scale"], 1e-5)?;

    // 5. LSQ+ parity
    let w_lsq_plus =
        topic::lsq_plus_quantize(&w["w_orig"], &w["scale_w"], &w["beta_w"], -8.0, 7.0)?;
    common::assert_max_diff(&w_lsq_plus, &w["w_lsq_plus"], 1e-5)?;

    // 6. QAT forward pass parity
    let (out_qat, _, _) = topic::qat_forward(
        &w["x_calib"],
        &w["w_orig"],
        &w["scale_w"],
        &w["alpha_x"],
        Some(&w["b_orig"]),
    )?;
    common::assert_max_diff(&out_qat, &w["out_qat"], 1e-5)?;

    Ok(())
}
