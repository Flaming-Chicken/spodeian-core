use candle_core::{safetensors::load, Device, Result};

use ai_candle::c08_learning_paradigms_rl_and_alignment::t03_semi_supervised_fixmatch_mean_teacher_and_nnpu_learning as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_03_semi_supervised_fixmatch_mean_teacher_and_nnpu_learning_parity() -> Result<()> {
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path("08_learning_paradigms_rl_and_alignment/01_self_semi_meta_continual_federated/03_semi_supervised_fixmatch_mean_teacher_and_nnpu_learning.safetensors"),
        &device,
    )?;

    // 1. Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // 2. Mean Teacher
    let mt_loss = topic::mean_teacher_consistency_loss(&w["student_logits"], &w["teacher_logits"])?;
    common::assert_max_diff(&mt_loss, &w["mt_loss_ref"], 1e-5)?;

    // 3. FixMatch
    let fm_loss = topic::fixmatch_thresholded_loss(&w["weak_logits"], &w["strong_logits"], 0.6)?;
    common::assert_max_diff(&fm_loss, &w["fixmatch_loss_ref"], 1e-5)?;

    // 4. MixMatch Sharpen
    let mm_sharp = topic::mixmatch_sharpen(&w["p_unsharpened"], 0.5)?;
    common::assert_max_diff(&mm_sharp, &w["mixmatch_sharpen_ref"], 1e-5)?;

    // 5. nnPU Risk
    let nnpu = topic::nnpu_risk(&w["g_pos"], &w["g_unlabeled"], 0.3)?;
    common::assert_max_diff(&nnpu, &w["nnpu_risk_ref"], 1e-5)?;

    Ok(())
}
