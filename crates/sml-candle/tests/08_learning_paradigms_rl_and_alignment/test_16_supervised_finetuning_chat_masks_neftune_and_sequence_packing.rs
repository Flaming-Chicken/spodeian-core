use candle_core::{safetensors::load, Device, Result};

use ai_candle::c08_learning_paradigms_rl_and_alignment::t16_supervised_finetuning_chat_masks_neftune_and_sequence_packing as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_16_supervised_finetuning_chat_masks_neftune_and_sequence_packing_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning/01_supervised_finetuning_chat_masks_neftune_and_sequence_packing.safetensors"),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. NEFTune Add Noise
    let noisy = topic::neftune_add_noise(&w["embeddings"], &w["noise"], 5.0)?;
    common::assert_max_diff(&noisy, &w["neftune_noisy_ref"], 1e-5)?;

    // 3. Masked Completion Cross-Entropy
    let loss =
        topic::masked_completion_cross_entropy(&w["logits"], &w["targets"], &w["loss_mask"], 1e-9)?;
    common::assert_max_diff(&loss, &w["masked_loss_ref"], 1e-5)?;

    // 4. Packed Block-Diagonal Causal Mask
    let mask = topic::packed_block_diagonal_causal_mask(&w["segment_ids"])?;
    common::assert_max_diff(&mask, &w["packed_mask_ref"], 1e-5)?;

    // 5. Packed Position IDs
    let pos = topic::packed_position_ids(&w["segment_ids"])?;
    common::assert_max_diff(
        &pos.to_dtype(candle_core::DType::F32)?,
        &w["packed_positions_ref"].to_dtype(candle_core::DType::F32)?,
        1e-5,
    )?;

    Ok(())
}
