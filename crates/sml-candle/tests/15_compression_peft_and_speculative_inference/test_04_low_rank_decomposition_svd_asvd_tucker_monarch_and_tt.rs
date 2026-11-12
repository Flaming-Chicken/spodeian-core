use candle_core::{safetensors::load, Device, Result};

use ai_candle::c15_compression_peft_and_speculative_inference::t04_low_rank_decomposition_svd_asvd_tucker_monarch_and_tt as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_04_low_rank_decomposition_svd_asvd_tucker_monarch_and_tt_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/04_low_rank_decomposition_svd_asvd_tucker_monarch_and_tt.safetensors"),
        &device,
    )?;

    // 1. Linear anchor parity
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Truncated SVD parity
    let (w_svd, s_svd) = topic::truncated_svd(&w["w"], 4)?;
    common::assert_max_diff(&w_svd, &w["w_svd"], 1e-5)?;
    common::assert_max_diff(&s_svd, &w["s_svd"], 1e-5)?;

    // 3. ASVD parity
    let (w_asvd, s_asvd) = topic::asvd_decompose(&w["w"], &w["act_scales"], 4)?;
    common::assert_max_diff(&w_asvd, &w["w_asvd"], 1e-5)?;
    common::assert_max_diff(&s_asvd, &w["s_asvd"], 1e-5)?;

    // 4. Monarch forward parity
    let y_monarch = topic::monarch_forward(&w["x"], &w["b1"], &w["b2"])?;
    common::assert_max_diff(&y_monarch, &w["y_monarch"], 1e-5)?;

    Ok(())
}
