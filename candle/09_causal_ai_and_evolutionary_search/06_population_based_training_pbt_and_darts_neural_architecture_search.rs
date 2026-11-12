//! Population Based Training PBT and DARTS Neural Architecture Search
//! in candle/09_causal_ai_and_evolutionary_search.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Population-Based Training exploit-and-explore step.
pub fn pbt_exploit_explore_step(
    scores: &Tensor,
    weights_pop: &Tensor,
    hyperparams_pop: &Tensor,
    perturb_factors: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let s_vec = scores.flatten_all()?.to_vec1::<f32>()?;
    let mut top_idx = 0;
    let mut bot_idx = 0;
    let mut max_s = f32::NEG_INFINITY;
    let mut min_s = f32::INFINITY;

    for (i, &s) in s_vec.iter().enumerate() {
        if s > max_s {
            max_s = s;
            top_idx = i;
        }
        if s < min_s {
            min_s = s;
            bot_idx = i;
        }
    }

    let mut w_mat = weights_pop.to_vec2::<f32>()?;
    let mut h_mat = hyperparams_pop.to_vec2::<f32>()?;
    let p_factors = perturb_factors.flatten_all()?.to_vec1::<f32>()?;

    w_mat[bot_idx] = w_mat[top_idx].clone();
    for m in 0..p_factors.len() {
        h_mat[bot_idx][m] = h_mat[top_idx][m] * p_factors[m];
    }

    let p = weights_pop.dim(0)?;
    let d = weights_pop.dim(1)?;
    let m = hyperparams_pop.dim(1)?;
    let dev = weights_pop.device();

    let mut w_flat = Vec::with_capacity(p * d);
    for row in w_mat {
        w_flat.extend(row);
    }
    let mut h_flat = Vec::with_capacity(p * m);
    for row in h_mat {
        h_flat.extend(row);
    }

    let w_tensor = Tensor::from_vec(w_flat, (p, d), dev)?;
    let h_tensor = Tensor::from_vec(h_flat, (p, m), dev)?;

    Ok((w_tensor, h_tensor))
}

/// DARTS continuous softmax relaxation mixed-operator forward pass.
pub fn darts_mixed_op_forward(
    alpha: &Tensor,
    candidate_outputs: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let softmax_weights = candle_nn::ops::softmax(alpha, 1)?;
    let w_exp = softmax_weights.unsqueeze(2)?.unsqueeze(3)?;
    let mixed_outputs = candidate_outputs.broadcast_mul(&w_exp)?.sum(1)?;
    Ok((mixed_outputs, softmax_weights))
}

/// Discretely decode DARTS architecture by selecting argmax operation per edge.
pub fn darts_discrete_decode(
    alpha: &Tensor,
    candidate_outputs: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let alpha_mat = alpha.to_vec2::<f32>()?;
    let e = alpha.dim(0)?;
    let mut selected_ops = Vec::with_capacity(e);
    let mut discrete_tensors = Vec::with_capacity(e);

    for (edge_idx, row) in alpha_mat.iter().enumerate() {
        let mut best_m = 0;
        let mut max_val = f32::NEG_INFINITY;
        for (m, &val) in row.iter().enumerate() {
            if val > max_val {
                max_val = val;
                best_m = m;
            }
        }
        selected_ops.push(best_m as f32);
        let edge_slice = candidate_outputs.get(edge_idx)?;
        let op_slice = edge_slice.get(best_m)?;
        discrete_tensors.push(op_slice);
    }

    let dev = alpha.device();
    let ops_tensor = Tensor::from_vec(selected_ops, (e,), dev)?;
    let disc_tensor = Tensor::stack(&discrete_tensors, 0)?;

    Ok((ops_tensor, disc_tensor))
}

/// Compute architecture distribution entropy: H(w) = -sum w log(w).
pub fn darts_architecture_entropy(softmax_weights: &Tensor, eps: f64) -> Result<Tensor> {
    let log_w = softmax_weights.affine(1.0, eps)?.log()?;
    softmax_weights.mul(&log_w)?.sum_all()?.neg()
}
