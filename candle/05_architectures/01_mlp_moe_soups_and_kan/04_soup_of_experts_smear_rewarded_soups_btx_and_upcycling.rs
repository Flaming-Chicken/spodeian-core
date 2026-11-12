//! Category 05 Topic 04: Soup of Experts, SMEAR, Rewarded Soups, BTX, and Sparse Upcycling.
//!
//! Provides native Candle implementations for:
//! 1. Soup of Experts (SMEAR) dynamic parameter merging before execution.
//! 2. Activation-Space MoE for empirical comparison.
//! 3. Rewarded Pareto Soups (static weight interpolation).

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

/// Soup of Experts (SMEAR) forward pass with dynamic parameter merging.
pub fn soup_of_experts_smear_forward(
    x: &Tensor,
    experts_w1: &Tensor,
    experts_b1: &Tensor,
    experts_w2: &Tensor,
    experts_b2: &Tensor,
    router_w: &Tensor,
    router_b: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (b_sz, _s_len, _d_model) = x.dims3()?;
    let num_experts = experts_w1.dim(0)?;

    // Mean pool across sequence dimension: [B, D]
    let x_pool = x.mean(1)?;
    let routing_logits = x_pool.matmul(router_w)?.broadcast_add(router_b)?;
    let routing_weights = softmax(&routing_logits, 1)?;
    let weights_vec = routing_weights.to_vec2::<f32>()?;

    let mut out_batches = Vec::with_capacity(b_sz);

    for b in 0..b_sz {
        let g = &weights_vec[b];
        let x_b = x.narrow(0, b, 1)?.squeeze(0)?; // [S, D]

        // Dynamically merge weights in parameter space
        let mut merged_w1 = (experts_w1.narrow(0, 0, 1)?.squeeze(0)? * (g[0] as f64))?;
        let mut merged_b1 = (experts_b1.narrow(0, 0, 1)?.squeeze(0)? * (g[0] as f64))?;
        let mut merged_w2 = (experts_w2.narrow(0, 0, 1)?.squeeze(0)? * (g[0] as f64))?;
        let mut merged_b2 = (experts_b2.narrow(0, 0, 1)?.squeeze(0)? * (g[0] as f64))?;

        for e in 1..num_experts {
            let scale = g[e] as f64;
            let w1_e = (experts_w1.narrow(0, e, 1)?.squeeze(0)? * scale)?;
            let b1_e = (experts_b1.narrow(0, e, 1)?.squeeze(0)? * scale)?;
            let w2_e = (experts_w2.narrow(0, e, 1)?.squeeze(0)? * scale)?;
            let b2_e = (experts_b2.narrow(0, e, 1)?.squeeze(0)? * scale)?;

            merged_w1 = (merged_w1 + w1_e)?;
            merged_b1 = (merged_b1 + b1_e)?;
            merged_w2 = (merged_w2 + w2_e)?;
            merged_b2 = (merged_b2 + b2_e)?;
        }

        // Single dense forward pass for sequence b
        let out_b = eval_expert(&x_b, &merged_w1, &merged_b1, &merged_w2, &merged_b2)?;
        out_batches.push(out_b);
    }

    let out_stacked = Tensor::stack(&out_batches, 0)?;
    Ok((out_stacked, routing_weights))
}

/// Conventional Activation-Space MoE forward pass for empirical comparison.
pub fn activation_moe_forward(
    x: &Tensor,
    experts_w1: &Tensor,
    experts_b1: &Tensor,
    experts_w2: &Tensor,
    experts_b2: &Tensor,
    router_w: &Tensor,
    router_b: &Tensor,
) -> Result<Tensor> {
    let (b_sz, s_len, d_model) = x.dims3()?;
    let num_experts = experts_w1.dim(0)?;

    let x_pool = x.mean(1)?;
    let routing_logits = x_pool.matmul(router_w)?.broadcast_add(router_b)?;
    let routing_weights = softmax(&routing_logits, 1)?;
    let weights_vec = routing_weights.to_vec2::<f32>()?;

    let mut out_batches = Vec::with_capacity(b_sz);

    for b in 0..b_sz {
        let g = &weights_vec[b];
        let x_b = x.narrow(0, b, 1)?.squeeze(0)?; // [S, D]
        let mut out_b = Tensor::zeros((s_len, d_model), x.dtype(), x.device())?;

        for e in 0..num_experts {
            let scale = g[e] as f64;
            let w1 = experts_w1.narrow(0, e, 1)?.squeeze(0)?;
            let b1 = experts_b1.narrow(0, e, 1)?.squeeze(0)?;
            let w2 = experts_w2.narrow(0, e, 1)?.squeeze(0)?;
            let b2 = experts_b2.narrow(0, e, 1)?.squeeze(0)?;

            let out_e = eval_expert(&x_b, &w1, &b1, &w2, &b2)?;
            let weighted_e = (out_e * scale)?;
            out_b = (out_b + weighted_e)?;
        }
        out_batches.push(out_b);
    }

    Tensor::stack(&out_batches, 0)
}

/// Evaluates static weight interpolation (Rewarded Pareto Soup) between expert 0 and 1.
pub fn evaluate_pareto_soup(
    x: &Tensor,
    experts_w1: &Tensor,
    experts_b1: &Tensor,
    experts_w2: &Tensor,
    experts_b2: &Tensor,
    alpha: f32,
) -> Result<Tensor> {
    let (b_sz, s_len, d_model) = x.dims3()?;
    let x_2d = x.contiguous()?.reshape((b_sz * s_len, d_model))?;

    let a = alpha as f64;
    let b = (1.0 - alpha) as f64;

    let w1_0 = (experts_w1.narrow(0, 0, 1)?.squeeze(0)? * a)?;
    let w1_1 = (experts_w1.narrow(0, 1, 1)?.squeeze(0)? * b)?;
    let w1 = (w1_0 + w1_1)?;

    let b1_0 = (experts_b1.narrow(0, 0, 1)?.squeeze(0)? * a)?;
    let b1_1 = (experts_b1.narrow(0, 1, 1)?.squeeze(0)? * b)?;
    let b1 = (b1_0 + b1_1)?;

    let w2_0 = (experts_w2.narrow(0, 0, 1)?.squeeze(0)? * a)?;
    let w2_1 = (experts_w2.narrow(0, 1, 1)?.squeeze(0)? * b)?;
    let w2 = (w2_0 + w2_1)?;

    let b2_0 = (experts_b2.narrow(0, 0, 1)?.squeeze(0)? * a)?;
    let b2_1 = (experts_b2.narrow(0, 1, 1)?.squeeze(0)? * b)?;
    let b2 = (b2_0 + b2_1)?;

    let out_2d = eval_expert(&x_2d, &w1, &b1, &w2, &b2)?;
    out_2d.reshape((b_sz, s_len, d_model))
}

/// Linear anchor forward pass for transitive cross-track parity verification.
pub fn linear_anchor_forward(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}
