use candle_core::{safetensors::load, Device, Result, Tensor};

use ai_candle::c15_compression_peft_and_speculative_inference::t03_llm_oneshot_pruning_wanda_and_sparsegpt as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_03_llm_oneshot_pruning_wanda_and_sparsegpt_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/03_llm_oneshot_pruning_wanda_and_sparsegpt.safetensors"),
        &device,
    )?;

    // 1. Linear anchor parity
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Activation norms
    let act_norms = topic::compute_activation_norms(&w["x"])?;
    common::assert_max_diff(&act_norms, &w["act_norms"], 1e-5)?;

    // 3. Wanda pruning
    let (w_wanda, mask_wanda) = topic::wanda_prune(&w["w"], &act_norms, 0.5)?;
    common::assert_max_diff(&w_wanda, &w["w_wanda"], 1e-5)?;
    common::assert_max_diff(&mask_wanda, &w["mask_wanda"], 1e-5)?;

    // 4. SparseGPT pruning
    let (w_sgpt, mask_sgpt) = topic::sparsegpt_prune(&w["w"], &w["h_inv"], 0.5)?;
    common::assert_max_diff(&w_sgpt, &w["w_sparsegpt"], 1e-5)?;
    common::assert_max_diff(&mask_sgpt, &w["mask_sparsegpt"], 1e-5)?;

    // 5. Error metrics
    let err_w = topic::compute_reconstruction_error(&w["x"], &w["w"], &w_wanda)?;
    let err_s = topic::compute_reconstruction_error(&w["x"], &w["w"], &w_sgpt)?;
    let err_vec = Tensor::from_vec(vec![err_w, err_s], (2,), &device)?;
    common::assert_max_diff(&err_vec, &w["err_metrics"], 1e-5)?;

    Ok(())
}
