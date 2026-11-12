//! Adaptive Compute Early Exit Token Merging and Frugalgpt Cascades
//! in candle/15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Early exit classification head evaluating confidence and exit decision.
pub fn early_exit_predict(
    hidden: &Tensor,
    head_weight: &Tensor,
    head_bias: &Tensor,
    threshold: f32,
) -> Result<(Tensor, Tensor, Tensor)> {
    let logits = hidden.matmul(head_weight)?.broadcast_add(head_bias)?;
    let probs = candle_nn::ops::softmax(&logits, 1)?;
    let (batch, classes) = probs.dims2()?;
    let p_vec = probs.flatten_all()?.to_vec1::<f32>()?;

    let mut conf_vec = vec![0.0f32; batch];
    let mut exit_vec = vec![0.0f32; batch];

    for b in 0..batch {
        let mut max_val = 0.0f32;
        for c in 0..classes {
            let val = p_vec[b * classes + c];
            if val > max_val {
                max_val = val;
            }
        }
        conf_vec[b] = max_val;
        exit_vec[b] = if max_val >= threshold { 1.0 } else { 0.0 };
    }

    let confidence = Tensor::from_vec(conf_vec, (batch,), hidden.device())?;
    let exit_mask = Tensor::from_vec(exit_vec, (batch,), hidden.device())?;
    Ok((probs, confidence, exit_mask))
}

/// Token Merging (ToMe) bipartite soft matching.
pub fn tome_bipartite_merge(
    x: &Tensor,
    r_merge: usize,
    token_sizes: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (batch, n, dim) = x.dims3()?;
    let x_vec = x.flatten_all()?.to_vec1::<f32>()?;
    let s_vec = token_sizes.flatten_all()?.to_vec1::<f32>()?;

    let len_a = (n + 1) / 2;
    let len_b = n / 2;
    let out_len = n - r_merge;

    let mut out_x = vec![0.0f32; batch * out_len * dim];
    let mut out_sizes = vec![0.0f32; batch * out_len];

    for b_idx in 0..batch {
        let b_offset_x = b_idx * n * dim;
        let b_offset_s = b_idx * n;

        // Partition tokens into A (even) and B (odd)
        let mut a_tokens = vec![vec![0.0f32; dim]; len_a];
        let mut b_tokens = vec![vec![0.0f32; dim]; len_b];
        let mut sizes_a = vec![0.0f32; len_a];
        let mut sizes_b = vec![0.0f32; len_b];

        for i in 0..len_a {
            let src = b_offset_x + (i * 2) * dim;
            a_tokens[i].copy_from_slice(&x_vec[src..src + dim]);
            sizes_a[i] = s_vec[b_offset_s + i * 2];
        }
        for j in 0..len_b {
            let src = b_offset_x + (j * 2 + 1) * dim;
            b_tokens[j].copy_from_slice(&x_vec[src..src + dim]);
            sizes_b[j] = s_vec[b_offset_s + j * 2 + 1];
        }

        // Pairwise cosine similarity between A and B
        let mut best_sim = vec![-1e9f32; len_a];
        let mut best_b = vec![0usize; len_a];

        for i in 0..len_a {
            let norm_a = a_tokens[i]
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt()
                .max(1e-8);
            for j in 0..len_b {
                let norm_b = b_tokens[j]
                    .iter()
                    .map(|x| x * x)
                    .sum::<f32>()
                    .sqrt()
                    .max(1e-8);
                let dot: f32 = (0..dim).map(|d| a_tokens[i][d] * b_tokens[j][d]).sum();
                let cos = dot / (norm_a * norm_b);
                if cos > best_sim[i] {
                    best_sim[i] = cos;
                    best_b[i] = j;
                }
            }
        }

        // Find top r_merge elements in A
        let mut a_indexed: Vec<(usize, f32)> = best_sim.into_iter().enumerate().collect();
        a_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let merged_a_indices: std::collections::HashSet<usize> = a_indexed
            .iter()
            .take(r_merge)
            .map(|&(idx, _)| idx)
            .collect();

        // Merge selected A tokens into target B
        for &a_idx in &merged_a_indices {
            let target_b = best_b[a_idx];
            let s_a = sizes_a[a_idx];
            let s_b = sizes_b[target_b];
            for d in 0..dim {
                b_tokens[target_b][d] =
                    (s_b * b_tokens[target_b][d] + s_a * a_tokens[a_idx][d]) / (s_a + s_b);
            }
            sizes_b[target_b] = s_a + s_b;
        }

        // Build output sequence: B followed by unmerged A
        let mut out_tokens: Vec<Vec<f32>> = b_tokens;
        let mut out_s: Vec<f32> = sizes_b;

        for a_idx in 0..len_a {
            if !merged_a_indices.contains(&a_idx) {
                out_tokens.push(a_tokens[a_idx].clone());
                out_s.push(sizes_a[a_idx]);
            }
        }

        // Write to flattened buffers
        for t in 0..out_len {
            for d in 0..dim {
                out_x[b_idx * out_len * dim + t * dim + d] = out_tokens[t][d];
            }
            out_sizes[b_idx * out_len + t] = out_s[t];
        }
    }

    let x_out = Tensor::from_vec(out_x, (batch, out_len, dim), x.device())?;
    let sizes_out = Tensor::from_vec(out_sizes, (batch, out_len), token_sizes.device())?;
    Ok((x_out, sizes_out))
}

/// FrugalGPT multi-tier model cascade router.
pub fn frugal_cascade_route(
    confidences: &Tensor,
    costs: &Tensor,
    thresholds: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (batch, num_models) = confidences.dims2()?;
    let conf_vec = confidences.flatten_all()?.to_vec1::<f32>()?;
    let cost_vec = costs.flatten_all()?.to_vec1::<f32>()?;
    let thresh_vec = thresholds.flatten_all()?.to_vec1::<f32>()?;

    let mut routed_idx = vec![(num_models - 1) as f32; batch];
    let mut total_cost = vec![0.0f32; batch];
    let mut resolved = vec![false; batch];

    for m in 0..num_models {
        for b in 0..batch {
            if !resolved[b] {
                total_cost[b] += cost_vec[m];
                if m < num_models - 1 {
                    if conf_vec[b * num_models + m] >= thresh_vec[m] {
                        routed_idx[b] = m as f32;
                        resolved[b] = true;
                    }
                } else {
                    routed_idx[b] = m as f32;
                    resolved[b] = true;
                }
            }
        }
    }

    let routed_tensor = Tensor::from_vec(routed_idx, (batch,), confidences.device())?;
    let cost_tensor = Tensor::from_vec(total_cost, (batch,), confidences.device())?;
    Ok((routed_tensor, cost_tensor))
}
