//! Knowledge Distillation Logit Feature and Minillm
//! in candle/15_compression_peft_and_speculative_inference/01_pruning_sparsity_lowrank_and_distillation.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Hinton logit-based knowledge distillation loss.
pub fn hinton_kd_loss(
    student_logits: &Tensor,
    teacher_logits: &Tensor,
    labels: &Tensor,
    temp: f64,
    alpha: f64,
) -> Result<(f32, f32)> {
    let (batch, _classes) = student_logits.dims2()?;
    let t_scale = 1.0 / temp;
    let s_scaled = (student_logits * t_scale)?;
    let t_scaled = (teacher_logits * t_scale)?;

    let p_s_log = candle_nn::ops::log_softmax(&s_scaled, 1)?;
    let p_t = candle_nn::ops::softmax(&t_scaled, 1)?;
    let p_t_log = candle_nn::ops::log_softmax(&t_scaled, 1)?;

    let kl_diff = (p_t_log.broadcast_sub(&p_s_log))?;
    let kl_div = (p_t.broadcast_mul(&kl_diff))?
        .sum_all()?
        .to_scalar::<f32>()?
        / (batch as f32);

    let s_log = candle_nn::ops::log_softmax(student_logits, 1)?;
    let ce_loss = candle_nn::loss::nll(&s_log, labels)?.to_scalar::<f32>()?;

    let total_loss =
        (1.0 - alpha as f32) * ce_loss + (alpha as f32) * (temp as f32 * temp as f32) * kl_div;
    Ok((total_loss, kl_div))
}

/// FitNets intermediate feature hint distillation MSE loss.
pub fn feature_distillation_loss(
    student_feat: &Tensor,
    teacher_feat: &Tensor,
    proj_weight: &Tensor,
) -> Result<f32> {
    let projected = student_feat.matmul(proj_weight)?;
    let diff = projected.broadcast_sub(teacher_feat)?;
    diff.sqr()?.mean_all()?.to_scalar::<f32>()
}

/// MiniLLM reverse KL divergence loss.
pub fn reverse_kl_loss(student_logits: &Tensor, teacher_logits: &Tensor, temp: f64) -> Result<f32> {
    let (batch, _classes) = student_logits.dims2()?;
    let t_scale = 1.0 / temp;
    let s_scaled = (student_logits * t_scale)?;
    let t_scaled = (teacher_logits * t_scale)?;

    let p_s = candle_nn::ops::softmax(&s_scaled, 1)?;
    let p_s_log = candle_nn::ops::log_softmax(&s_scaled, 1)?;
    let p_t_log = candle_nn::ops::log_softmax(&t_scaled, 1)?;

    let kl_diff = (p_s_log.broadcast_sub(&p_t_log))?;
    let rev_kl = (p_s.broadcast_mul(&kl_diff))?
        .sum_all()?
        .to_scalar::<f32>()?
        / (batch as f32);
    Ok(rev_kl)
}
