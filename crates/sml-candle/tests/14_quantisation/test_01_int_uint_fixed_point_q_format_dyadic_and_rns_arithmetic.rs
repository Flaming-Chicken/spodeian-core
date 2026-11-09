use candle_core::{safetensors::load, Device, Result};

use ai_candle::c14_quantisation::t01_int_uint_fixed_point_q_format_dyadic_and_rns_arithmetic as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_01_int_uint_fixed_point_q_format_dyadic_and_rns_arithmetic_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(common::get_weights_path("14_quantisation/01_foundations_formats_and_observers/01_int_uint_fixed_point_q_format_dyadic_and_rns_arithmetic.safetensors"), &device)?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Affine INT8
    let (aff_hat, aff_q, aff_s, aff_z) = topic::affine_quantize_dequantize(&w["raw_relu"], 8)?;
    common::assert_max_diff(&aff_hat, &w["affine_hat"], 1e-5)?;
    common::assert_max_diff(&aff_q, &w["affine_q"], 1e-5)?;
    let ref_aff_s = w["affine_s"].to_scalar::<f32>()? as f64;
    let ref_aff_z = w["affine_z"].to_scalar::<f32>()? as f64;
    assert!((aff_s - ref_aff_s).abs() < 1e-5);
    assert!((aff_z - ref_aff_z).abs() < 1e-5);

    // 3. Symmetric INT8 (restricted vs full)
    let (sym_restr_hat, sym_restr_q, sym_restr_s) =
        topic::symmetric_quantize_dequantize(&w["raw_sym"], 8, false)?;
    common::assert_max_diff(&sym_restr_hat, &w["sym_hat_restr"], 1e-5)?;
    common::assert_max_diff(&sym_restr_q, &w["sym_q_restr"], 1e-5)?;
    let ref_sym_restr_s = w["sym_s_restr"].to_scalar::<f32>()? as f64;
    assert!((sym_restr_s - ref_sym_restr_s).abs() < 1e-5);

    let (sym_full_hat, sym_full_q, sym_full_s) =
        topic::symmetric_quantize_dequantize(&w["raw_sym"], 8, true)?;
    common::assert_max_diff(&sym_full_hat, &w["sym_hat_full"], 1e-5)?;
    common::assert_max_diff(&sym_full_q, &w["sym_q_full"], 1e-5)?;
    let ref_sym_full_s = w["sym_s_full"].to_scalar::<f32>()? as f64;
    assert!((sym_full_s - ref_sym_full_s).abs() < 1e-5);

    // 4. Fixed-Point Q4.4
    let qmn = topic::fixed_point_qmn(&w["raw_fp"], 4, 4)?;
    common::assert_max_diff(&qmn, &w["qmn_hat"], 1e-5)?;

    // 5. Dyadic Rescaling
    let m0 = w["m0"].to_scalar::<f32>()? as i64;
    let shift = w["shift"].to_scalar::<f32>()? as i64;
    let dyadic_out = topic::dyadic_rescale_tensor(&w["acc_int"], m0, shift)?;
    common::assert_max_diff(&dyadic_out, &w["dyadic_out"], 1e-5)?;

    // 6. RNS Multiplication & CRT
    let moduli = [7i64, 11i64, 13i64];
    let rns_out = topic::rns_multiply_and_crt(&w["rns_a"], &w["rns_b"], &moduli)?;
    common::assert_max_diff(&rns_out, &w["rns_out"], 1e-5)?;

    Ok(())
}
