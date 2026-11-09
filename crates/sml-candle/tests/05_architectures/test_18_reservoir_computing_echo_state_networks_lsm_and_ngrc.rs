use candle_core::{safetensors::load, Device, Result};

use ai_candle::c05_architectures::t18_reservoir_computing_echo_state_networks_lsm_and_ngrc as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_18_reservoir_computing_echo_state_networks_lsm_and_ngrc_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("05_architectures/03_recurrent_ssm_and_memory/07_reservoir_computing_echo_state_networks_lsm_and_ngrc.safetensors"),
        &device,
    )?;

    // 1. Linear anchor parity
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. ESN forward parity
    let alpha = w["alpha"].to_vec1::<f32>()?[0] as f64;
    let (seq_out, final_h) = topic::esn_forward(
        &w["x_seq"],
        &w["w_in"],
        &w["w_res"],
        &w["b_res"],
        alpha,
        &w["w_out"],
        &w["b_out"],
    )?;
    common::assert_max_diff(&seq_out, &w["seq_out"], 1e-5)?;
    common::assert_max_diff(&final_h, &w["final_h"], 1e-5)?;

    // 3. NG-RC features parity
    let feat = topic::ngrc_features(&w["x_curr"], &w["x_prev"])?;
    common::assert_max_diff(&feat, &w["ngrc_feat"], 1e-5)?;

    Ok(())
}
