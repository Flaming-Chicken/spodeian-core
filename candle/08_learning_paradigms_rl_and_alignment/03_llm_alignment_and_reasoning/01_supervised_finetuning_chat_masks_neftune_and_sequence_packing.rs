//! Supervised Finetuning Chat Masks Neftune and Sequence Packing
//! in candle/08_learning_paradigms_rl_and_alignment/03_llm_alignment_and_reasoning.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// NEFTune: Noisy Embedding Instruction Fine-Tuning.
pub fn neftune_add_noise(embeddings: &Tensor, noise: &Tensor, alpha: f32) -> Result<Tensor> {
    let dims = embeddings.dims();
    let seq_len = dims[1];
    let d_dim = dims[2];
    let scale = alpha / ((seq_len * d_dim) as f32).sqrt();

    let scaled_noise = (noise * (scale as f64))?;
    embeddings + scaled_noise
}

/// Masked completion cross-entropy loss for conversational SFT.
pub fn masked_completion_cross_entropy(
    logits: &Tensor,
    targets: &Tensor,
    loss_mask: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let logit_vec = logits.to_vec3::<f32>()?;
    let target_vec = targets.to_vec2::<i64>()?;
    let mask_vec = loss_mask.to_vec2::<f32>()?;

    let b = logit_vec.len();
    let l = logit_vec[0].len();
    let v = logit_vec[0][0].len();

    let mut total_masked_nll = 0.0f32;
    let mut total_mask = 0.0f32;

    for bi in 0..b {
        for li in 0..l {
            let mut max_v = f32::NEG_INFINITY;
            for vi in 0..v {
                if logit_vec[bi][li][vi] > max_v {
                    max_v = logit_vec[bi][li][vi];
                }
            }

            let mut sum_exp = 0.0f32;
            for vi in 0..v {
                sum_exp += (logit_vec[bi][li][vi] - max_v).exp();
            }
            let log_sum_exp = max_v + sum_exp.ln();

            let target_idx = target_vec[bi][li] as usize;
            let log_p = logit_vec[bi][li][target_idx] - log_sum_exp;
            let nll = -log_p;
            let m = mask_vec[bi][li];

            total_masked_nll += m * nll;
            total_mask += m;
        }
    }

    Tensor::new(total_masked_nll / (total_mask + eps), logits.device())
}

/// Sequence packing block-diagonal causal attention mask.
pub fn packed_block_diagonal_causal_mask(segment_ids: &Tensor) -> Result<Tensor> {
    let seg = segment_ids.to_vec1::<i64>()?;
    let l = seg.len();
    let mut mask = Vec::with_capacity(l * l);

    for i in 0..l {
        for j in 0..l {
            let val = if j <= i && seg[i] == seg[j] {
                1.0f32
            } else {
                0.0f32
            };
            mask.push(val);
        }
    }

    Tensor::from_vec(mask, (l, l), segment_ids.device())
}

/// Sequence packing position IDs resetting per sub-sequence.
pub fn packed_position_ids(segment_ids: &Tensor) -> Result<Tensor> {
    let seg = segment_ids.to_vec1::<i64>()?;
    let l = seg.len();
    let mut pos_ids = Vec::with_capacity(l);
    let mut cur = 0i64;

    for i in 0..l {
        if i > 0 && seg[i] != seg[i - 1] {
            cur = 0;
        }
        pos_ids.push(cur);
        cur += 1;
    }

    Tensor::from_vec(pos_ids, (l,), segment_ids.device())
}
