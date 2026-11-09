use candle_core::{safetensors::load, Device, Result, Tensor};

use ai_candle::c15_compression_peft_and_speculative_inference::t01_unstructured_structured_shortgpt_and_lottery_ticket as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_01_unstructured_structured_shortgpt_and_lottery_ticket_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/01_unstructured_structured_shortgpt_and_lottery_ticket.safetensors"),
        &device,
    )?;

    // 1. Linear anchor parity
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Unstructured magnitude pruning
    let (w_unstr, mask_unstr) = topic::unstructured_magnitude_prune(&w["w"], 0.5)?;
    common::assert_max_diff(&w_unstr, &w["w_unstructured"], 1e-5)?;
    common::assert_max_diff(&mask_unstr, &w["mask_unstructured"], 1e-5)?;

    // 3. Structured row pruning
    let (w_str, indices_str) = topic::structured_row_prune(&w["w"], 0.75)?;
    common::assert_max_diff(&w_str, &w["w_structured"], 1e-5)?;
    common::assert_max_diff(&indices_str, &w["indices_structured"], 1e-5)?;

    // 4. ShortGPT Block Influence
    let bi_red = topic::compute_block_influence(&w["x_in"], &w["x_out_red"])?;
    let bi_act = topic::compute_block_influence(&w["x_in"], &w["x_out_act"])?;
    let bi_scores = Tensor::from_vec(vec![bi_red, bi_act], (2,), &device)?;
    common::assert_max_diff(&bi_scores, &w["bi_scores"], 1e-5)?;

    // 5. Lottery ticket subnetwork
    let (w_sub, mask_lottery) = topic::lottery_ticket_rewind(&w["w"], &w["w_trained"], 0.5)?;
    common::assert_max_diff(&w_sub, &w["w_lottery_subnetwork"], 1e-5)?;
    common::assert_max_diff(&mask_lottery, &w["mask_lottery"], 1e-5)?;

    Ok(())
}
