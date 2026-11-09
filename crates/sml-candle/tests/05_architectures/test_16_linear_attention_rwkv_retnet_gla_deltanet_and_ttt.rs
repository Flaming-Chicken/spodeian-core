use candle_core::{safetensors::load, Device, Result};

use ai_candle::c05_architectures::t16_linear_attention_rwkv_retnet_gla_deltanet_and_ttt as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_16_linear_attention_rwkv_retnet_gla_deltanet_and_ttt_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("05_architectures/03_recurrent_ssm_and_memory/05_linear_attention_rwkv_retnet_gla_deltanet_and_ttt.safetensors"),
        &device,
    )?;

    // 1. Linear anchor parity
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. RetNet recurrent and parallel parity
    let gamma = w["gamma"].to_vec1::<f32>()?[0] as f64;
    let (ret_rec_out, ret_s) = topic::retnet_recurrent_forward(&w["q"], &w["k"], &w["v"], gamma)?;
    common::assert_max_diff(&ret_rec_out, &w["ret_recurrent_out"], 1e-5)?;
    common::assert_max_diff(&ret_s, &w["ret_final_s"], 1e-5)?;

    let ret_par_out = topic::retnet_parallel_forward(&w["q"], &w["k"], &w["v"], gamma)?;
    common::assert_max_diff(&ret_par_out, &w["ret_parallel_out"], 1e-5)?;

    // 3. DeltaNet forward parity
    let (delta_out, delta_s) = topic::deltanet_forward(&w["q"], &w["k"], &w["v"], &w["beta"])?;
    common::assert_max_diff(&delta_out, &w["delta_out"], 1e-5)?;
    common::assert_max_diff(&delta_s, &w["delta_final_s"], 1e-5)?;

    Ok(())
}
