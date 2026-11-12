use candle_core::{safetensors::load, Device, Result};

use ai_candle::c14_quantisation::t26_quantization_method_selection_matrix_and_head_to_head_benchmark as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_26_quantization_method_selection_matrix_and_head_to_head_benchmark_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "14_quantisation/05_kv_attention_optim_and_benchmarks/06_quantization_method_selection_matrix_and_head_to_head_benchmark.safetensors",
        ),
        &device,
    )?;

    // Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    let weight = &w["w"];
    let x = &w["x"];

    // 1. RTN INT8
    let w_rtn8 = topic::rtn_quantize(weight, 8)?;
    common::assert_max_diff(&w_rtn8, &w["w_rtn8"], 1e-5)?;
    let y_rtn8 = x.matmul(&w_rtn8)?;
    common::assert_max_diff(&y_rtn8, &w["y_rtn8"], 1e-5)?;

    // 2. RTN INT4
    let w_rtn4 = topic::rtn_quantize(weight, 4)?;
    common::assert_max_diff(&w_rtn4, &w["w_rtn4"], 1e-5)?;
    let y_rtn4 = x.matmul(&w_rtn4)?;
    common::assert_max_diff(&y_rtn4, &w["y_rtn4"], 1e-5)?;

    // 3. AWQ INT4
    let w_awq4 = topic::awq_quantize(weight, x, 4)?;
    common::assert_max_diff(&w_awq4, &w["w_awq4"], 1e-5)?;
    let y_awq4 = x.matmul(&w_awq4)?;
    common::assert_max_diff(&y_awq4, &w["y_awq4"], 1e-5)?;

    // 4. BitNet b1.58
    let w_bitnet = topic::bitnet_quantize(weight)?;
    common::assert_max_diff(&w_bitnet, &w["w_bitnet"], 1e-5)?;
    let y_bitnet = x.matmul(&w_bitnet)?;
    common::assert_max_diff(&y_bitnet, &w["y_bitnet"], 1e-5)?;

    // 5. NF4
    let w_nf4 = topic::nf4_quantize(weight)?;
    common::assert_max_diff(&w_nf4, &w["w_nf4"], 1e-5)?;
    let y_nf4 = x.matmul(&w_nf4)?;
    common::assert_max_diff(&y_nf4, &w["y_nf4"], 1e-5)?;

    // Ground truth activation
    let y_true = x.matmul(weight)?;
    common::assert_max_diff(&y_true, &w["y_true"], 1e-5)?;

    // SQNR parity verification
    let methods = [
        (&w_rtn8, &y_rtn8),
        (&w_rtn4, &y_rtn4),
        (&w_awq4, &y_awq4),
        (&w_bitnet, &y_bitnet),
        (&w_nf4, &y_nf4),
    ];

    let w_sqnrs_ref = w["weight_sqnrs"].to_vec1::<f32>()?;
    let act_sqnrs_ref = w["act_sqnrs"].to_vec1::<f32>()?;

    for (i, (w_q, y_q)) in methods.iter().enumerate() {
        let w_sqnr = topic::compute_sqnr(weight, &(weight - *w_q)?)?;
        let act_sqnr = topic::compute_sqnr(&y_true, &(&y_true - *y_q)?)?;
        assert!(
            (w_sqnr - w_sqnrs_ref[i]).abs() < 1e-2,
            "Method {i} weight SQNR mismatch: computed {w_sqnr}, expected {}",
            w_sqnrs_ref[i]
        );
        assert!(
            (act_sqnr - act_sqnrs_ref[i]).abs() < 1e-2,
            "Method {i} act SQNR mismatch: computed {act_sqnr}, expected {}",
            act_sqnrs_ref[i]
        );
    }

    Ok(())
}
