//! Tree Speculative Medusa Eagle GoT and Jacobi
//! in candle/15_compression_peft_and_speculative_inference/02_adaptive_kv_speculative_and_merging.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Medusa multi-head forward prediction.
pub fn medusa_heads_predict(
    hidden: &Tensor,
    head_weights: &Tensor,
    head_biases: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (batch, _dim) = hidden.dims2()?;
    let (num_heads, _, vocab) = head_weights.dims3()?;

    let mut logits_list = Vec::with_capacity(num_heads);
    let mut top_tokens_vec = vec![0.0f32; batch * num_heads];

    for k in 0..num_heads {
        let w_k = head_weights.get(k)?;
        let b_k = head_biases.get(k)?;
        let logits_k = hidden.matmul(&w_k)?.broadcast_add(&b_k)?;
        let k_vec = logits_k.flatten_all()?.to_vec1::<f32>()?;

        for b in 0..batch {
            let mut max_val = f32::NEG_INFINITY;
            let mut max_idx = 0usize;
            for v in 0..vocab {
                let val = k_vec[b * vocab + v];
                if val > max_val {
                    max_val = val;
                    max_idx = v;
                }
            }
            top_tokens_vec[b * num_heads + k] = max_idx as f32;
        }

        logits_list.push(logits_k.unsqueeze(1)?);
    }

    let logits = Tensor::cat(&logits_list, 1)?;
    let top_tokens = Tensor::from_vec(top_tokens_vec, (batch, num_heads), hidden.device())?;
    Ok((logits, top_tokens))
}

/// Construct 2D causal tree attention mask from tree node ancestry.
pub fn build_tree_attention_mask(parent_indices: &Tensor) -> Result<Tensor> {
    let parents = parent_indices.flatten_all()?.to_vec1::<f32>()?;
    let num_nodes = parents.len();
    let mut mask_vec = vec![-1e9f32; num_nodes * num_nodes];

    for i in 0..num_nodes {
        let mut curr = i;
        let mut visited = std::collections::HashSet::new();
        while !visited.contains(&curr) {
            mask_vec[i * num_nodes + curr] = 0.0;
            visited.insert(curr);
            if curr == 0 {
                break;
            }
            curr = parents[curr] as usize;
        }
    }

    Tensor::from_vec(mask_vec, (num_nodes, num_nodes), parent_indices.device())
}

/// Find the longest continuous matching path in speculative candidate tree.
pub fn tree_longest_accepted_path(
    candidate_tokens: &Tensor,
    target_tokens: &Tensor,
    parent_indices: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let cands = candidate_tokens.flatten_all()?.to_vec1::<f32>()?;
    let targets = target_tokens.flatten_all()?.to_vec1::<f32>()?;
    let parents = parent_indices.flatten_all()?.to_vec1::<f32>()?;
    let num_nodes = cands.len();

    let mut accepted = vec![false; num_nodes];
    accepted[0] = true;

    for i in 1..num_nodes {
        let p = parents[i] as usize;
        if accepted[p] && (cands[i] == targets[i]) {
            accepted[i] = true;
        }
    }

    let mut depths = vec![0usize; num_nodes];
    let mut best_len = 0usize;
    let mut best_node = 0usize;

    for i in 0..num_nodes {
        if accepted[i] {
            depths[i] = if i == 0 {
                1
            } else {
                depths[parents[i] as usize] + 1
            };
            if depths[i] > best_len {
                best_len = depths[i];
                best_node = i;
            }
        }
    }

    let mut path_mask_vec = vec![0.0f32; num_nodes];
    let mut curr = best_node;
    loop {
        path_mask_vec[curr] = 1.0;
        if curr == 0 {
            break;
        }
        curr = parents[curr] as usize;
    }

    let path_mask = Tensor::from_vec(path_mask_vec, (num_nodes,), candidate_tokens.device())?;
    let path_length = Tensor::new(best_len as f32, candidate_tokens.device())?;
    Ok((path_mask, path_length))
}

/// Jacobi fixed-point non-autoregressive iteration step.
pub fn jacobi_lookahead_step(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (batch, seq_len, dim) = x.dims3()?;
    let x_flat = x.reshape((batch * seq_len, dim))?;
    let projected = x_flat.matmul(weight)?.broadcast_add(bias)?;
    let x_next_flat = projected.tanh()?;
    let x_next = x_next_flat.reshape((batch, seq_len, dim))?;

    let diff = (x_next.broadcast_sub(x))?.abs()?;
    let diff_vec = diff.flatten_all()?.to_vec1::<f32>()?;

    let items_per_batch = seq_len * dim;
    let mut max_res = vec![0.0f32; batch];
    for b in 0..batch {
        let b_slice = &diff_vec[b * items_per_batch..(b + 1) * items_per_batch];
        max_res[b] = b_slice.iter().cloned().fold(0.0f32, f32::max);
    }

    let residual = Tensor::from_vec(max_res, (batch,), x.device())?;
    Ok((x_next, residual))
}
