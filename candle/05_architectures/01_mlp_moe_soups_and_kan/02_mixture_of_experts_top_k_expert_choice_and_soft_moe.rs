//! Category 05 Topic 02: Mixture of Experts (Top-K, Expert Choice, and Soft MoE).
//!
//! Provides native Candle implementations for:
//! 1. Sparse Top-K MoE with auxiliary load-balancing loss and router z-loss.
//! 2. Expert Choice MoE with inverted token routing for perfect load balance.
//! 3. Soft MoE with soft token dispatch and combine across expert slots.

use candle_core::{Result, Tensor};

/// Numerically stable softmax along a specified dimension.
pub fn softmax(xs: &Tensor, dim: usize) -> Result<Tensor> {
    let max = xs.max_keepdim(dim)?;
    let diff = xs.broadcast_sub(&max)?;
    let exp = diff.exp()?;
    let sum = exp.sum_keepdim(dim)?;
    exp.broadcast_div(&sum)
}

/// Evaluates a 2-layer MLP expert on a 2D tensor: [T, D] -> [T, D].
pub fn eval_expert(
    x: &Tensor,
    w1: &Tensor,
    b1: &Tensor,
    w2: &Tensor,
    b2: &Tensor,
) -> Result<Tensor> {
    let h = x.matmul(w1)?.broadcast_add(b1)?.gelu_erf()?;
    h.matmul(w2)?.broadcast_add(b2)
}

/// Sparse Top-K Mixture of Experts forward pass with auxiliary loss and z-loss.
pub fn topk_moe_forward(
    x: &Tensor,
    router_w: &Tensor,
    router_b: &Tensor,
    experts_w1: &Tensor,
    experts_b1: &Tensor,
    experts_w2: &Tensor,
    experts_b2: &Tensor,
    top_k: usize,
) -> Result<(Tensor, f32, f32)> {
    let (b_sz, s_len, d_model) = x.dims3()?;
    let num_tokens = b_sz * s_len;
    let num_experts = router_w.dim(1)?;
    let x_flat = x.contiguous()?.reshape((num_tokens, d_model))?;

    // Router logits: [T, E]
    let router_logits = x_flat.matmul(router_w)?.broadcast_add(router_b)?;
    let router_probs = softmax(&router_logits, 1)?;

    // Pull router logits to host to perform deterministic top-k dispatch
    let logits_vec: Vec<f32> = router_logits
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let probs_vec: Vec<f32> = router_probs
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();

    let mut out_flat_vec = vec![0.0f32; num_tokens * d_model];
    let mut f_count = vec![0.0f32; num_experts];

    for t in 0..num_tokens {
        let x_t = x_flat.narrow(0, t, 1)?;
        let t_logits_start = t * num_experts;
        let mut expert_scores: Vec<(usize, f32)> = (0..num_experts)
            .map(|e| (e, logits_vec[t_logits_start + e]))
            .collect();
        // Sort descending by logit
        expert_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Compute softmax over top-k
        let topk_scores: Vec<f32> = expert_scores.iter().take(top_k).map(|(_, s)| *s).collect();
        let max_score = topk_scores
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = topk_scores.iter().map(|s| (s - max_score).exp()).collect();
        let sum_exp: f32 = exps.iter().sum();
        let weights: Vec<f32> = exps.iter().map(|e| e / sum_exp).collect();

        for k_idx in 0..top_k {
            let (e_idx, _) = expert_scores[k_idx];
            let w = weights[k_idx];
            f_count[e_idx] += 1.0;

            let w1 = experts_w1.narrow(0, e_idx, 1)?.squeeze(0)?;
            let b1 = experts_b1.narrow(0, e_idx, 1)?.squeeze(0)?;
            let w2 = experts_w2.narrow(0, e_idx, 1)?.squeeze(0)?;
            let b2 = experts_b2.narrow(0, e_idx, 1)?.squeeze(0)?;

            let expert_out = eval_expert(&x_t, &w1, &b1, &w2, &b2)?;
            let out_vec = expert_out.to_vec2::<f32>()?[0].clone();

            for d in 0..d_model {
                out_flat_vec[t * d_model + d] += w * out_vec[d];
            }
        }
    }

    // Aux load balancing loss: E * sum_i(f_i * P_i)
    let mut mean_probs = vec![0.0f32; num_experts];
    for t in 0..num_tokens {
        for e in 0..num_experts {
            mean_probs[e] += probs_vec[t * num_experts + e];
        }
    }
    for e in 0..num_experts {
        mean_probs[e] /= num_tokens as f32;
    }

    let mut aux_loss = 0.0f32;
    for e in 0..num_experts {
        let f_i = f_count[e] / (num_tokens as f32);
        aux_loss += f_i * mean_probs[e];
    }
    aux_loss *= num_experts as f32;

    // Router Z-loss: (1/T) * sum_t (logsumexp(H_t))^2
    let mut z_loss = 0.0f32;
    for t in 0..num_tokens {
        let row_start = t * num_experts;
        let row = &logits_vec[row_start..row_start + num_experts];
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum_e: f32 = row.iter().map(|v| (v - max_val).exp()).sum();
        let lse = sum_e.ln() + max_val;
        z_loss += lse * lse;
    }
    z_loss /= num_tokens as f32;

    let out_flat = Tensor::from_vec(out_flat_vec, (num_tokens, d_model), x.device())?;
    let out = out_flat.reshape((b_sz, s_len, d_model))?;
    Ok((out, aux_loss, z_loss))
}

/// Expert Choice MoE forward pass with inverted routing.
pub fn expert_choice_moe_forward(
    x: &Tensor,
    router_w: &Tensor,
    router_b: &Tensor,
    experts_w1: &Tensor,
    experts_b1: &Tensor,
    experts_w2: &Tensor,
    experts_b2: &Tensor,
    capacity: usize,
) -> Result<Tensor> {
    let (b_sz, s_len, d_model) = x.dims3()?;
    let num_tokens = b_sz * s_len;
    let num_experts = router_w.dim(1)?;
    let x_flat = x.contiguous()?.reshape((num_tokens, d_model))?;

    let router_logits = x_flat.matmul(router_w)?.broadcast_add(router_b)?;
    let router_probs = softmax(&router_logits, 1)?;
    let probs_vec: Vec<f32> = router_probs
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();

    let mut out_flat_vec = vec![0.0f32; num_tokens * d_model];

    for e in 0..num_experts {
        // Collect scores for all tokens with respect to expert e
        let mut token_scores: Vec<(usize, f32)> = (0..num_tokens)
            .map(|t| (t, probs_vec[t * num_experts + e]))
            .collect();
        token_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let w1 = experts_w1.narrow(0, e, 1)?.squeeze(0)?;
        let b1 = experts_b1.narrow(0, e, 1)?.squeeze(0)?;
        let w2 = experts_w2.narrow(0, e, 1)?.squeeze(0)?;
        let b2 = experts_b2.narrow(0, e, 1)?.squeeze(0)?;

        for c_idx in 0..capacity {
            let (t, score) = token_scores[c_idx];
            let x_t = x_flat.narrow(0, t, 1)?;
            let expert_out = eval_expert(&x_t, &w1, &b1, &w2, &b2)?;
            let out_vec = expert_out.to_vec2::<f32>()?[0].clone();

            for d in 0..d_model {
                out_flat_vec[t * d_model + d] += score * out_vec[d];
            }
        }
    }

    let out_flat = Tensor::from_vec(out_flat_vec, (num_tokens, d_model), x.device())?;
    out_flat.reshape((b_sz, s_len, d_model))
}

/// Soft MoE forward pass with soft dispatch and combine.
pub fn soft_moe_forward(
    x: &Tensor,
    soft_disp_w: &Tensor,
    soft_disp_b: &Tensor,
    soft_comb_w: &Tensor,
    soft_comb_b: &Tensor,
    experts_w1: &Tensor,
    experts_b1: &Tensor,
    experts_w2: &Tensor,
    experts_b2: &Tensor,
    slots_per_expert: usize,
) -> Result<Tensor> {
    let (b_sz, s_len, d_model) = x.dims3()?;
    let num_tokens = b_sz * s_len;
    let total_slots = soft_disp_w.dim(1)?;
    let x_flat = x.contiguous()?.reshape((num_tokens, d_model))?;

    // Dispatch: softmax across tokens (dim 0)
    let disp_logits = x_flat.matmul(soft_disp_w)?.broadcast_add(soft_disp_b)?;
    let phi = softmax(&disp_logits, 0)?;

    // Combine: softmax across slots (dim 1)
    let comb_logits = x_flat.matmul(soft_comb_w)?.broadcast_add(soft_comb_b)?;
    let psi = softmax(&comb_logits, 1)?;

    // Slot inputs: [M, D] = phi^T @ x_flat
    let slot_inputs = phi.t()?.matmul(&x_flat)?;

    let mut slot_outputs_vec = Vec::with_capacity(total_slots);
    for j in 0..total_slots {
        let e = j / slots_per_expert;
        let slot_in = slot_inputs.narrow(0, j, 1)?;
        let w1 = experts_w1.narrow(0, e, 1)?.squeeze(0)?;
        let b1 = experts_b1.narrow(0, e, 1)?.squeeze(0)?;
        let w2 = experts_w2.narrow(0, e, 1)?.squeeze(0)?;
        let b2 = experts_b2.narrow(0, e, 1)?.squeeze(0)?;
        let slot_out = eval_expert(&slot_in, &w1, &b1, &w2, &b2)?;
        slot_outputs_vec.push(slot_out);
    }

    let slot_outputs = Tensor::cat(&slot_outputs_vec, 0)?;
    let out_flat = psi.matmul(&slot_outputs)?;
    out_flat.reshape((b_sz, s_len, d_model))
}

/// Linear anchor forward pass for transitive cross-track parity verification.
pub fn linear_anchor_forward(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}
