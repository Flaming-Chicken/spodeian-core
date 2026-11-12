use candle_core::{safetensors::load, Device, Result};

use ai_candle::c14_quantisation::t18_binary_sub1bit_and_ternary_bnn_xnor_billm_onebit_twn_ttq as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_18_binary_sub1bit_and_ternary_bnn_xnor_billm_onebit_twn_ttq_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("14_quantisation/04_vector_lattice_qat_and_lowbit/04_binary_sub1bit_and_ternary_bnn_xnor_billm_onebit_twn_ttq.safetensors"),
        &device,
    )?;

    // 1. Linear anchor parity
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. BNN parity
    let (w_bnn, b_signs, b_alpha) = topic::bnn_xnor_quantize(&w["w_orig"])?;
    common::assert_max_diff(&w_bnn, &w["w_bnn"], 1e-5)?;
    common::assert_max_diff(&b_signs, &w["b_signs"], 1e-5)?;
    common::assert_max_diff(&b_alpha, &w["b_alpha"], 1e-5)?;

    // 3. TWN parity
    let (w_twn, t_ternary, t_alpha) = topic::twn_quantize(&w["w_orig"])?;
    common::assert_max_diff(&w_twn, &w["w_twn"], 1e-5)?;
    common::assert_max_diff(&t_ternary, &w["t_ternary"], 1e-5)?;
    common::assert_max_diff(&t_alpha, &w["t_alpha"], 1e-5)?;

    // 4. TTQ parity
    let (w_ttq, ttq_t, w_pos, w_neg) = topic::ttq_quantize(&w["w_orig"], 0.35)?;
    common::assert_max_diff(&w_ttq, &w["w_ttq"], 1e-5)?;
    common::assert_max_diff(&ttq_t, &w["ttq_t"], 1e-5)?;
    common::assert_max_diff(&w_pos, &w["w_pos"], 1e-5)?;
    common::assert_max_diff(&w_neg, &w["w_neg"], 1e-5)?;

    // 5. OneBit parity
    let (w_onebit, onebit_signs, onebit_g, onebit_h) = topic::onebit_quantize(&w["w_orig"])?;
    common::assert_max_diff(&w_onebit, &w["w_onebit"], 1e-5)?;
    common::assert_max_diff(&onebit_signs, &w["onebit_signs"], 1e-5)?;
    common::assert_max_diff(&onebit_g, &w["onebit_g"], 1e-5)?;
    common::assert_max_diff(&onebit_h, &w["onebit_h"], 1e-5)?;

    Ok(())
}
