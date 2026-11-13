use candle_core::{Device, Result, Tensor, safetensors::load};

use ai_candle::c15_compression_peft_and_speculative_inference::t05_knowledge_distillation_logit_feature_and_minillm as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_05_knowledge_distillation_logit_feature_and_minillm_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation/05_knowledge_distillation_logit_feature_and_minillm.safetensors",
        ),
        &device,
    )?;

    // 1. Linear anchor parity
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Hinton KD loss parity
    let (total_kd, fwd_kl) = topic::hinton_kd_loss(
        &w["student_logits"],
        &w["teacher_logits"],
        &w["labels"],
        2.0,
        0.5,
    )?;

    // 3. Reverse KL divergence parity
    let rev_kl = topic::reverse_kl_loss(&w["student_logits"], &w["teacher_logits"], 2.0)?;

    // 4. Feature map distillation parity
    let feat_loss = topic::feature_distillation_loss(
        &w["student_feat"],
        &w["teacher_feat"],
        &w["proj_weight"],
    )?;

    let calculated_metrics =
        Tensor::from_vec(vec![total_kd, fwd_kl, rev_kl, feat_loss], (4,), &device)?;
    common::assert_max_diff(&calculated_metrics, &w["loss_metrics"], 1e-5)?;

    Ok(())
}
