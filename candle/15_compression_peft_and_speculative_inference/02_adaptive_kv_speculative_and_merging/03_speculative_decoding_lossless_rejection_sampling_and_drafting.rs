//! Speculative Decoding Lossless Rejection Sampling and Drafting
//! in candle/15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Execute lossless speculative rejection sampling across gamma candidate tokens.
pub fn speculative_rejection_step(
    target_probs: &Tensor,
    draft_probs: &Tensor,
    draft_tokens: &Tensor,
    uniform_rands: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (batch, gamma, vocab) = target_probs.dims3()?;
    let t_vec = target_probs.flatten_all()?.to_vec1::<f32>()?;
    let d_vec = draft_probs.flatten_all()?.to_vec1::<f32>()?;
    let tok_vec = draft_tokens.flatten_all()?.to_vec1::<f32>()?;
    let u_vec = uniform_rands.flatten_all()?.to_vec1::<f32>()?;

    let mut accepted_mask_vec = vec![0.0f32; batch * gamma];
    let mut num_accepted_vec = vec![0.0f32; batch];
    let mut residual_probs_vec = vec![0.0f32; batch * vocab];

    for b in 0..batch {
        let b_offset_gamma = b * gamma;
        let mut rejected_pos = -1isize;
        let mut acc_count = 0.0f32;

        for i in 0..gamma {
            let t_id = tok_vec[b_offset_gamma + i] as usize;
            let prob_offset = (b * gamma + i) * vocab;
            let p_val = t_vec[prob_offset + t_id];
            let q_val = d_vec[prob_offset + t_id];
            let ratio = (p_val / q_val.max(1e-8)).min(1.0);

            if u_vec[b_offset_gamma + i] <= ratio {
                accepted_mask_vec[b_offset_gamma + i] = 1.0;
                acc_count += 1.0;
            } else {
                rejected_pos = i as isize;
                break;
            }
        }

        num_accepted_vec[b] = acc_count;

        if rejected_pos >= 0 {
            let pos = rejected_pos as usize;
            let prob_offset = (b * gamma + pos) * vocab;
            let mut diff = vec![0.0f32; vocab];
            let mut norm = 0.0f32;

            for v in 0..vocab {
                let d = (t_vec[prob_offset + v] - d_vec[prob_offset + v]).max(0.0);
                diff[v] = d;
                norm += d;
            }

            let res_offset = b * vocab;
            if norm > 1e-8 {
                for v in 0..vocab {
                    residual_probs_vec[res_offset + v] = diff[v] / norm;
                }
            } else {
                for v in 0..vocab {
                    residual_probs_vec[res_offset + v] = t_vec[prob_offset + v];
                }
            }
        } else {
            let last_prob_offset = (b * gamma + (gamma - 1)) * vocab;
            let res_offset = b * vocab;
            for v in 0..vocab {
                residual_probs_vec[res_offset + v] = t_vec[last_prob_offset + v];
            }
        }
    }

    let accepted_mask = Tensor::from_vec(accepted_mask_vec, (batch, gamma), target_probs.device())?;
    let num_accepted = Tensor::from_vec(num_accepted_vec, (batch,), target_probs.device())?;
    let residual_probs =
        Tensor::from_vec(residual_probs_vec, (batch, vocab), target_probs.device())?;

    Ok((accepted_mask, num_accepted, residual_probs))
}

/// Zero-parameter speculative drafting via n-gram prompt prefix lookup.
pub fn prompt_lookup_ngram_draft(
    input_ids: &Tensor,
    ngram_size: usize,
    gamma: usize,
) -> Result<Tensor> {
    let ids_vec = input_ids.flatten_all()?.to_vec1::<f32>()?;
    let seq_len = ids_vec.len();

    if seq_len < ngram_size + 1 {
        return Tensor::from_vec(vec![0.0f32; gamma], (gamma,), input_ids.device());
    }

    let suffix = &ids_vec[seq_len - ngram_size..seq_len];
    let mut best_pos = -1isize;

    for i in (0..=seq_len - ngram_size - 1).rev() {
        let candidate = &ids_vec[i..i + ngram_size];
        if candidate == suffix {
            best_pos = (i + ngram_size) as isize;
            break;
        }
    }

    let mut draft = vec![0.0f32; gamma];
    if best_pos >= 0 {
        let pos = best_pos as usize;
        for k in 0..gamma {
            if pos + k < seq_len {
                draft[k] = ids_vec[pos + k];
            }
        }
    }

    Tensor::from_vec(draft, (gamma,), input_ids.device())
}
