//! Category 05 Topic 03: DeepSeekMoE, Auxiliary-Loss-Free Load Balancing, and Mixture of Depths.
//!
//! Provides native Candle implementations for:
//! 1. DeepSeekMoE with Fine-Grained + Shared Experts.
//! 2. Auxiliary-Loss-Free Load Balancing via dynamic router bias updates.
//! 3. Mixture of Depths (MoD) with dynamic token-level capacity allocation.

use candle_core::{Result, Tensor};

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

/// Numerically stable sigmoid function.
pub fn sigmoid(x: &Tensor) -> Result<Tensor> {
    let neg_x = x.neg()?;
    let exp_neg = neg_x.exp()?;
    let one = Tensor::ones_like(&exp_neg)?;
    let denom = (one + exp_neg)?;
    Tensor::ones_like(&denom)?.div(&denom)
}

/// DeepSeekMoE forward pass with shared experts and aux-loss-free top-k routing.
pub fn deepseek_moe_forward(
    x: &Tensor,
    router_w: &Tensor,
    router_b: &Tensor,
    b_dyn: &Tensor,
    shared_w1: &Tensor,
    shared_b1: &Tensor,
    shared_w2: &Tensor,
    shared_b2: &Tensor,
    routed_w1: &Tensor,
    routed_b1: &Tensor,
    routed_w2: &Tensor,
    routed_b2: &Tensor,
    top_k: usize,
) -> Result<(Tensor, Vec<f32>)> {
    let (b_sz, s_len, d_model) = x.dims3()?;
    let num_tokens = b_sz * s_len;
    let num_shared = shared_w1.dim(0)?;
    let num_routed = routed_w1.dim(0)?;
    let x_flat = x.contiguous()?.reshape((num_tokens, d_model))?;

    let mut out_flat = Tensor::zeros((num_tokens, d_model), x.dtype(), x.device())?;

    // 1. Unconditionally evaluate all shared experts
    for s in 0..num_shared {
        let w1 = shared_w1.narrow(0, s, 1)?.squeeze(0)?;
        let b1 = shared_b1.narrow(0, s, 1)?.squeeze(0)?;
        let w2 = shared_w2.narrow(0, s, 1)?.squeeze(0)?;
        let b2 = shared_b2.narrow(0, s, 1)?.squeeze(0)?;
        let shared_out = eval_expert(&x_flat, &w1, &b1, &w2, &b2)?;
        out_flat = (out_flat + shared_out)?;
    }

    // 2. Router logits and aux-loss-free top-k selection
    let router_logits = x_flat.matmul(router_w)?.broadcast_add(router_b)?;
    let biased_logits = router_logits.broadcast_add(b_dyn)?;

    let raw_logits_vec: Vec<f32> = router_logits
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let biased_vec: Vec<f32> = biased_logits
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();

    let mut out_flat_vec = out_flat
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect::<Vec<f32>>();
    let mut expert_counts = vec![0.0f32; num_routed];

    for t in 0..num_tokens {
        let x_t = x_flat.narrow(0, t, 1)?;
        let t_start = t * num_routed;

        // Top-k index selection using BIASED scores
        let mut expert_scores: Vec<(usize, f32)> = (0..num_routed)
            .map(|e| (e, biased_vec[t_start + e]))
            .collect();
        expert_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Softmax weights computed using UNBIASED raw logits
        let topk_raw: Vec<f32> = expert_scores
            .iter()
            .take(top_k)
            .map(|(e, _)| raw_logits_vec[t_start + e])
            .collect();
        let max_raw = topk_raw.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = topk_raw.iter().map(|s| (s - max_raw).exp()).collect();
        let sum_exp: f32 = exps.iter().sum();
        let weights: Vec<f32> = exps.iter().map(|e| e / sum_exp).collect();

        for k_idx in 0..top_k {
            let (e_idx, _) = expert_scores[k_idx];
            let w = weights[k_idx];
            expert_counts[e_idx] += 1.0;

            let w1 = routed_w1.narrow(0, e_idx, 1)?.squeeze(0)?;
            let b1 = routed_b1.narrow(0, e_idx, 1)?.squeeze(0)?;
            let w2 = routed_w2.narrow(0, e_idx, 1)?.squeeze(0)?;
            let b2 = routed_b2.narrow(0, e_idx, 1)?.squeeze(0)?;

            let expert_out = eval_expert(&x_t, &w1, &b1, &w2, &b2)?;
            let out_vec = expert_out.to_vec2::<f32>()?[0].clone();

            for d in 0..d_model {
                out_flat_vec[t * d_model + d] += w * out_vec[d];
            }
        }
    }

    let result_flat = Tensor::from_vec(out_flat_vec, (num_tokens, d_model), x.device())?;
    let result = result_flat.reshape((b_sz, s_len, d_model))?;
    Ok((result, expert_counts))
}

/// Updates dynamic routing bias vector outside autograd.
pub fn update_dynamic_bias(b_dyn: &[f32], counts: &[f32], gamma: f32) -> Vec<f32> {
    let avg_count: f32 = counts.iter().sum::<f32>() / (counts.len() as f32);
    b_dyn
        .iter()
        .zip(counts.iter())
        .map(|(&b, &c)| {
            let diff = avg_count - c;
            let sgn = if diff > 0.0 {
                1.0
            } else if diff < 0.0 {
                -1.0
            } else {
                0.0
            };
            b + gamma * sgn
        })
        .collect()
}

/// Mixture of Depths (MoD) forward pass with fixed token capacity.
pub fn mixture_of_depths_forward(
    x: &Tensor,
    mod_w: &Tensor,
    mod_b: &Tensor,
    block_w1: &Tensor,
    block_b1: &Tensor,
    block_w2: &Tensor,
    block_b2: &Tensor,
    capacity: usize,
) -> Result<Tensor> {
    let (b_sz, s_len, d_model) = x.dims3()?;
    let x_2d = x.contiguous()?.reshape((b_sz * s_len, d_model))?;
    let logits_2d = x_2d.matmul(mod_w)?.broadcast_add(mod_b)?;
    let router_logits = logits_2d.reshape((b_sz, s_len))?;
    let router_weights = sigmoid(&router_logits)?;
    let weights_vec: Vec<Vec<f32>> = router_weights.to_vec2::<f32>()?;

    let mut out_vec = x.to_vec3::<f32>()?;

    for b in 0..b_sz {
        let mut token_scores: Vec<(usize, f32)> =
            (0..s_len).map(|s| (s, weights_vec[b][s])).collect();
        token_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for c_idx in 0..capacity {
            let (s, gate) = token_scores[c_idx];
            let token_in = x.narrow(0, b, 1)?.narrow(1, s, 1)?.squeeze(0)?;
            let block_out = eval_expert(&token_in, block_w1, block_b1, block_w2, block_b2)?;
            let out_vals = block_out.to_vec2::<f32>()?[0].clone();

            for d in 0..d_model {
                out_vec[b][s][d] += gate * out_vals[d];
            }
        }
    }

    let flat: Vec<f32> = out_vec.into_iter().flatten().flatten().collect();
    Tensor::from_vec(flat, (b_sz, s_len, d_model), x.device())
}

/// Linear anchor forward pass for transitive cross-track parity verification.
pub fn linear_anchor_forward(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}
