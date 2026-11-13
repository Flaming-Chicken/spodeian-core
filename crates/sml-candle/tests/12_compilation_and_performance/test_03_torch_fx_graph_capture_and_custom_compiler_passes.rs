use candle_core::{Device, Result, safetensors::load};

use ai_candle::c12_compilation_and_performance::t03_torch_fx_graph_capture_and_custom_compiler_passes as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_03_torch_fx_graph_capture_and_custom_compiler_passes_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "12_compilation_and_performance/03_torch_fx_graph_capture_and_custom_compiler_passes.safetensors",
        ),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Linear-BatchNorm folding
    let (w_fold, b_fold) = topic::algebraic_linear_bn_folding(
        &w["lin_w"],
        &w["lin_b"],
        &w["bn_mean"],
        &w["bn_var"],
        &w["bn_gamma"],
        &w["bn_beta"],
        1e-5,
    )?;
    common::assert_max_diff(&w_fold, &w["w_fold_ref"], 1e-5)?;
    common::assert_max_diff(&b_fold, &w["b_fold_ref"], 1e-5)?;

    let folded_out = w["lin_x"].matmul(&w_fold)?.broadcast_add(&b_fold)?;
    common::assert_max_diff(&folded_out, &w["folded_out_ref"], 1e-5)?;

    let unfolded_out = topic::sequential_linear_bn(
        &w["lin_x"],
        &w["lin_w"],
        &w["lin_b"],
        &w["bn_mean"],
        &w["bn_var"],
        &w["bn_gamma"],
        &w["bn_beta"],
        1e-5,
    )?;
    common::assert_max_diff(&unfolded_out, &folded_out, 1e-5)?;

    // 3. Parallel GEMM fusion
    let (w_fused, b_fused) =
        topic::parallel_gemm_fusion(&w["w1"], &w["b1"], &w["w2"], &w["b2"], &w["w3"], &w["b3"])?;
    common::assert_max_diff(&w_fused, &w["w_fused_ref"], 1e-5)?;
    common::assert_max_diff(&b_fused, &w["b_fused_ref"], 1e-5)?;

    let fused_out = w["lin_x"].matmul(&w_fused)?.broadcast_add(&b_fused)?;
    common::assert_max_diff(&fused_out, &w["fused_out_ref"], 1e-5)?;

    Ok(())
}
