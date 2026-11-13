use candle_core::{Device, Result, safetensors::load};

use ai_candle::c05_architectures::t17_memory_architectures_ntm_pkm_peer_infini_and_titans as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_17_memory_architectures_ntm_pkm_peer_infini_and_titans_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "05_architectures/03_recurrent_ssm_and_memory/06_memory_architectures_ntm_pkm_peer_infini_and_titans.safetensors",
        ),
        &device,
    )?;

    // 1. Linear anchor parity
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. NTM Content Addressing
    let (read_vec, ntm_weights) =
        topic::ntm_content_addressing_read(&w["memory"], &w["key"], &w["beta"])?;
    common::assert_max_diff(&read_vec, &w["read_vec"], 1e-5)?;
    common::assert_max_diff(&ntm_weights, &w["ntm_weights"], 1e-5)?;

    // 3. Infini-Attention
    let (a_mem, m_next, z_next) =
        topic::infini_memory_step(&w["m_prev"], &w["z_prev"], &w["q"], &w["k"], &w["v"])?;
    common::assert_max_diff(&a_mem, &w["a_mem"], 1e-5)?;
    common::assert_max_diff(&m_next, &w["m_next"], 1e-5)?;
    common::assert_max_diff(&z_next, &w["z_next"], 1e-5)?;

    Ok(())
}
