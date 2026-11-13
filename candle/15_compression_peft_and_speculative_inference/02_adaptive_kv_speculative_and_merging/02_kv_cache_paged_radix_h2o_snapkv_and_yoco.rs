//! KV Cache Paged Radix H2O Snapkv and Yoco
//! in candle/15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Gather non-contiguous KV cache pages and compute scaled dot-product attention.
pub fn paged_kv_gather_and_attend(
    query: &Tensor,
    key_blocks: &Tensor,
    value_blocks: &Tensor,
    block_table: &Tensor,
    context_len: usize,
    block_size: usize,
) -> Result<(Tensor, Tensor)> {
    let (batch, _, dim) = query.dims3()?;
    let scale = (1.0f64 / (dim as f64).sqrt()) as f32;

    let k_vec = key_blocks.flatten_all()?.to_vec1::<f32>()?;
    let v_vec = value_blocks.flatten_all()?.to_vec1::<f32>()?;
    let q_vec = query.flatten_all()?.to_vec1::<f32>()?;
    let bt_vec = block_table.flatten_all()?.to_vec1::<f32>()?;
    let (_, max_blocks) = block_table.dims2()?;

    let mut out_vec = vec![0.0f32; batch * dim];
    let mut weights_vec = vec![0.0f32; batch * context_len];

    for b in 0..batch {
        // Gather keys and values for context_len tokens
        let mut k_gathered = vec![0.0f32; context_len * dim];
        let mut v_gathered = vec![0.0f32; context_len * dim];

        for t in 0..context_len {
            let logical_block = t / block_size;
            let block_offset = t % block_size;
            let phys_block = bt_vec[b * max_blocks + logical_block] as usize;

            let src_idx = (phys_block * block_size + block_offset) * dim;
            let dst_idx = t * dim;
            k_gathered[dst_idx..dst_idx + dim].copy_from_slice(&k_vec[src_idx..src_idx + dim]);
            v_gathered[dst_idx..dst_idx + dim].copy_from_slice(&v_vec[src_idx..src_idx + dim]);
        }

        // Attention dot products: q_b dot k_t
        let q_offset = b * dim;
        let mut scores = vec![0.0f32; context_len];
        for t in 0..context_len {
            let mut dot = 0.0f32;
            for d in 0..dim {
                dot += q_vec[q_offset + d] * k_gathered[t * dim + d];
            }
            scores[t] = dot * scale;
        }

        // Softmax over context_len
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        let mut probs = vec![0.0f32; context_len];
        for t in 0..context_len {
            let p = (scores[t] - max_score).exp();
            probs[t] = p;
            sum_exp += p;
        }
        for t in 0..context_len {
            probs[t] /= sum_exp;
            weights_vec[b * context_len + t] = probs[t];
        }

        // Context = probs @ V
        for d in 0..dim {
            let mut val = 0.0f32;
            for t in 0..context_len {
                val += probs[t] * v_gathered[t * dim + d];
            }
            out_vec[b * dim + d] = val;
        }
    }

    let attn_out = Tensor::from_vec(out_vec, (batch, 1, dim), query.device())?;
    let attn_weights = Tensor::from_vec(weights_vec, (batch, 1, context_len), query.device())?;
    Ok((attn_out, attn_weights))
}

/// Heavy Hitter Oracle (H2O) cache retention mask computation.
pub fn h2o_evict_mask(
    cumulative_scores: &Tensor,
    local_window: usize,
    heavy_budget: usize,
) -> Result<Tensor> {
    let (batch, seq_len) = cumulative_scores.dims2()?;
    let scores_vec = cumulative_scores.flatten_all()?.to_vec1::<f32>()?;
    let mut mask_vec = vec![0.0f32; batch * seq_len];

    let recent_start = seq_len.saturating_sub(local_window);

    for b in 0..batch {
        let b_offset = b * seq_len;
        // Keep recent tokens
        for t in recent_start..seq_len {
            mask_vec[b_offset + t] = 1.0;
        }

        // Candidate historical tokens
        let cand_len = recent_start;
        if cand_len > 0 && heavy_budget > 0 {
            let mut indexed_scores: Vec<(usize, f32)> = (0..cand_len)
                .map(|t| (t, scores_vec[b_offset + t]))
                .collect();
            indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            for &(idx, _) in indexed_scores.iter().take(heavy_budget) {
                mask_vec[b_offset + idx] = 1.0;
            }
        }
    }

    Tensor::from_vec(mask_vec, (batch, seq_len), cumulative_scores.device())
}

/// SnapKV: Observation window pooling for prompt KV cache compression.
pub fn snapkv_select_mask(
    prompt_attention: &Tensor,
    _obs_window: usize,
    budget: usize,
) -> Result<Tensor> {
    let (batch, obs_win, prompt_len) = prompt_attention.dims3()?;
    let p_vec = prompt_attention.flatten_all()?.to_vec1::<f32>()?;
    let mut mask_vec = vec![0.0f32; batch * prompt_len];

    for b in 0..batch {
        let mut importance = vec![0.0f32; prompt_len];
        for t in 0..prompt_len {
            let mut sum_p = 0.0f32;
            for w in 0..obs_win {
                let idx = b * (obs_win * prompt_len) + w * prompt_len + t;
                sum_p += p_vec[idx];
            }
            importance[t] = sum_p / (obs_win as f32);
        }

        let mut indexed: Vec<(usize, f32)> = importance.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let k = budget.min(prompt_len);
        for &(idx, _) in indexed.iter().take(k) {
            mask_vec[b * prompt_len + idx] = 1.0;
        }
    }

    Tensor::from_vec(mask_vec, (batch, prompt_len), prompt_attention.device())
}

/// You Only Cache Once (YOCO) cross-decoder forward step.
pub fn yoco_cross_layer_forward(
    h_mid: &Tensor,
    k_shared: &Tensor,
    v_shared: &Tensor,
    w_q: &Tensor,
    w_out: &Tensor,
) -> Result<Tensor> {
    let (b, s, dim) = h_mid.dims3()?;
    let scale = 1.0f64 / (dim as f64).sqrt();

    let q = h_mid
        .reshape((b * s, dim))?
        .matmul(w_q)?
        .reshape((b, s, dim))?;
    let k_t = k_shared.transpose(1, 2)?;
    let scores = (q.matmul(&k_t)? * scale)?;
    let attn = candle_nn::ops::softmax(&scores, 2)?;
    let context = attn.matmul(v_shared)?;
    let projected = context
        .reshape((b * s, dim))?
        .matmul(w_out)?
        .reshape((b, s, dim))?;
    h_mid.broadcast_add(&projected)
}
