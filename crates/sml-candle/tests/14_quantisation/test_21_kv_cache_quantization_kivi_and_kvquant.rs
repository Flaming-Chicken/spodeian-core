use candle_core::{safetensors::load, Device, Result};

use ai_candle::c14_quantisation::t21_kv_cache_quantization_kivi_and_kvquant as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_21_kv_cache_quantization_kivi_and_kvquant_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("14_quantisation/05_kv_attention_optim_and_benchmarks/01_kv_cache_quantization_kivi_and_kvquant.safetensors"),
        &device,
    )?;

    // 1. Linear anchor parity
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Key per-channel quantization parity
    let (k_deq, k_int, k_scale, k_zero) = topic::quantize_k_per_channel(&w["k"], 2)?;
    common::assert_max_diff(&k_deq, &w["k_deq"], 1e-5)?;
    common::assert_max_diff(&k_int, &w["k_int"], 1e-5)?;
    common::assert_max_diff(&k_scale, &w["k_scale"], 1e-5)?;
    common::assert_max_diff(&k_zero, &w["k_zero"], 1e-5)?;

    // 3. Value per-token quantization parity
    let (v_deq, v_int, v_scale, v_zero) = topic::quantize_v_per_token(&w["v"], 2)?;
    common::assert_max_diff(&v_deq, &w["v_deq"], 1e-5)?;
    common::assert_max_diff(&v_int, &w["v_int"], 1e-5)?;
    common::assert_max_diff(&v_scale, &w["v_scale"], 1e-5)?;
    common::assert_max_diff(&v_zero, &w["v_zero"], 1e-5)?;

    // 4. KIVI Attention Decoding parity
    let (out_kivi, k_eff, v_eff) = topic::kivi_attention_decode(&w["q"], &w["k"], &w["v"], 4, 2)?;
    common::assert_max_diff(&out_kivi, &w["out_kivi"], 1e-5)?;
    common::assert_max_diff(&k_eff, &w["k_eff"], 1e-5)?;
    common::assert_max_diff(&v_eff, &w["v_eff"], 1e-5)?;

    Ok(())
}
