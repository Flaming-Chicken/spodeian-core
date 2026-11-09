use candle_core::{safetensors::load, Device, Result};

use ai_candle::c14_quantisation::t19_bitnet_b1_58_native_ternary_transformers as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_19_bitnet_b1_58_native_ternary_transformers_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("14_quantisation/04_vector_lattice_qat_and_lowbit/05_bitnet_b1_58_native_ternary_transformers.safetensors"),
        &device,
    )?;

    // 1. Linear anchor parity
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. RMSNorm parity
    let x_norm = topic::rms_norm(&w["x_raw"], &w["rms_weight"], 1e-5)?;
    common::assert_max_diff(&x_norm, &w["x_norm"], 1e-5)?;

    // 3. Activation quantization parity
    let (x_q, x_int8, gamma) = topic::activation_quant_bitnet(&w["x_norm"], 127.0, 1e-5)?;
    common::assert_max_diff(&x_q, &w["x_q"], 1e-5)?;
    common::assert_max_diff(&x_int8, &w["x_int8"], 1e-5)?;
    common::assert_max_diff(&gamma, &w["gamma"], 1e-5)?;

    // 4. Weight quantization parity
    let (w_q, w_ternary, beta) = topic::weight_quant_bitnet(&w["w_orig"], 1e-5)?;
    common::assert_max_diff(&w_q, &w["w_q"], 1e-5)?;
    common::assert_max_diff(&w_ternary, &w["w_ternary"], 1e-5)?;
    common::assert_max_diff(&beta, &w["beta"], 1e-5)?;

    // 5. Integer accumulator parity
    let int_acc = x_int8.matmul(&w_ternary)?;
    common::assert_max_diff(&int_acc, &w["integer_acc"], 1e-5)?;

    // 6. BitLinear forward pass parity
    let (out_bitlinear, _, _) = topic::bitlinear_forward(
        &w["x_raw"],
        &w["w_orig"],
        &w["rms_weight"],
        Some(&w["b_orig"]),
    )?;
    common::assert_max_diff(&out_bitlinear, &w["out_bitlinear"], 1e-5)?;

    Ok(())
}
