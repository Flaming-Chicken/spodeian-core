use candle_core::{safetensors::load, Device, Result};

use ai_candle::c05_architectures::t15_mamba_selective_ssm_and_mamba2_ssd as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_15_mamba_selective_ssm_and_mamba2_ssd_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("05_architectures/03_recurrent_ssm_and_memory/04_mamba_selective_ssm_and_mamba2_ssd.safetensors"),
        &device,
    )?;

    // 1. Anchor parity
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Mamba Selective SSM forward parity
    let (seq_out, final_h) = topic::mamba_selective_forward(
        &w["x_seq"],
        &w["a_log"],
        &w["w_b"],
        &w["w_c"],
        &w["w_delta"],
        &w["b_delta"],
        &w["d_skip"],
    )?;
    common::assert_max_diff(&seq_out, &w["seq_out"], 1e-5)?;
    common::assert_max_diff(&final_h, &w["final_h"], 1e-5)?;

    // 3. Mamba-2 SSD mask parity
    let l_matrix = topic::mamba2_ssd_mask(&w["delta_scalar"], &w["a_scalar"])?;
    common::assert_max_diff(&l_matrix, &w["l_matrix"], 1e-5)?;

    Ok(())
}
