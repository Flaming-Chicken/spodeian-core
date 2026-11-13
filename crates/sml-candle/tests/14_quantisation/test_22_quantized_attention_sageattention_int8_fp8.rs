use candle_core::{Device, Result, safetensors::load};

use sml_candle::c14_quantisation::t22_quantized_attention_sageattention_int8_fp8 as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_22_quantized_attention_sageattention_int8_fp8_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "14_quantisation/05_kv_attention_optim_and_benchmarks/02_quantized_attention_sageattention_int8_fp8.safetensors",
        ),
        &device,
    )?;

    // 1. Linear anchor parity
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Key-mean smoothing parity
    let (k_smoothed, k_mean) = topic::key_mean_smoothing(&w["k"])?;
    common::assert_max_diff(&k_smoothed, &w["k_smoothed"], 1e-5)?;
    common::assert_max_diff(&k_mean, &w["k_mean"], 1e-5)?;

    // 3. INT8 quantization parity
    let (q_deq, q_int8, s_q) = topic::quantize_symmetric_int8(&w["q"], 1e-7)?;
    common::assert_max_diff(&q_deq, &w["q_deq"], 1e-5)?;
    common::assert_max_diff(&q_int8, &w["q_int8"], 1e-5)?;
    common::assert_max_diff(&s_q, &w["s_q"], 1e-5)?;

    let (k_deq, k_int8, s_k) = topic::quantize_symmetric_int8(&w["k_smoothed"], 1e-7)?;
    common::assert_max_diff(&k_deq, &w["k_deq"], 1e-5)?;
    common::assert_max_diff(&k_int8, &w["k_int8"], 1e-5)?;
    common::assert_max_diff(&s_k, &w["s_k"], 1e-5)?;

    let (v_deq, v_int8, s_v) = topic::quantize_symmetric_int8(&w["v"], 1e-7)?;
    common::assert_max_diff(&v_deq, &w["v_deq"], 1e-5)?;
    common::assert_max_diff(&v_int8, &w["v_int8"], 1e-5)?;
    common::assert_max_diff(&s_v, &w["s_v"], 1e-5)?;

    // 4. SageAttention forward pass parity
    let (out_sage, scores_sage, probs_sage, int_acc, _) =
        topic::sage_attention(&w["q"], &w["k"], &w["v"])?;
    common::assert_max_diff(&int_acc, &w["int_acc"], 1e-5)?;
    common::assert_max_diff(&scores_sage, &w["scores_sage"], 1e-5)?;
    common::assert_max_diff(&probs_sage, &w["probs_sage"], 1e-5)?;
    common::assert_max_diff(&out_sage, &w["out_sage"], 1e-5)?;

    Ok(())
}
