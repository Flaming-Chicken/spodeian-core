use candle_core::{safetensors::load, Device, Result};

use ai_candle::c12_compilation_and_performance::t07_profiling_roofline_analysis_cuda_graphs_and_fused_optimizers as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_07_profiling_roofline_analysis_cuda_graphs_and_fused_optimizers_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("12_compilation_and_performance/07_profiling_roofline_analysis_cuda_graphs_and_fused_optimizers.safetensors"),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Fused AdamW Step
    let (adam_w, adam_m, adam_v) = topic::fused_adamw_step(
        &w["adam_w"],
        &w["adam_g"],
        &w["adam_m"],
        &w["adam_v"],
        5,
        1e-3,
        0.9,
        0.999,
        1e-2,
        1e-8,
    )?;
    common::assert_max_diff(&adam_w, &w["adam_w_ref"], 1e-5)?;
    common::assert_max_diff(&adam_m, &w["adam_m_ref"], 1e-5)?;
    common::assert_max_diff(&adam_v, &w["adam_v_ref"], 1e-5)?;

    // 3. Roofline Metrics
    let (intensity, ridge, attainable) = topic::roofline_metrics(1e12, 1e9, 312.0, 2.039);
    assert!((intensity - 1000.0).abs() < 1e-5);
    assert!((ridge - (312.0 / 2.039)).abs() < 1e-4);
    assert!((attainable - 312.0).abs() < 1e-5);

    Ok(())
}
