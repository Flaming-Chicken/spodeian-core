//! Expert Parallelism and 3D/4D Parallel Composition in candle/16_distributed_and_scalable.

use candle_core::{Device, Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Maps global rank indices to 3D Cartesian coordinates (dp_idx, pp_idx, tp_idx).
pub fn compute_3d_mesh_coordinates(
    dp: usize,
    pp: usize,
    tp: usize,
    device: &Device,
) -> Result<Tensor> {
    let world_size = dp * pp * tp;
    let mut coords = Vec::with_capacity(world_size * 3);

    for r in 0..world_size {
        let dp_idx = (r / (pp * tp)) as i64;
        let rem = r % (pp * tp);
        let pp_idx = (rem / tp) as i64;
        let tp_idx = (rem % tp) as i64;
        coords.push(dp_idx);
        coords.push(pp_idx);
        coords.push(tp_idx);
    }

    Tensor::from_slice(&coords, (world_size, 3), device)
}

/// Expert Parallelism with Top-2 routing, dispatch mask, and combined output.
pub fn expert_parallel_forward(
    x: &Tensor,
    w_gate: &Tensor,
    w_experts: &Tensor,
    ep_world_size: usize,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let (num_tokens, d_model) = x.dims2()?;
    let num_experts = w_experts.dim(0)?;
    let experts_per_rank = num_experts / ep_world_size;
    let device = x.device();

    let gate_logits = x.matmul(w_gate)?; // [T, E]

    let mut top2_indices_vec = Vec::with_capacity(num_tokens * 2);
    let mut top2_probs_vec = Vec::with_capacity(num_tokens * 2);
    let mut moe_out_vec = vec![0.0f32; num_tokens * d_model];

    for t in 0..num_tokens {
        let logits_t = gate_logits.get(t)?.to_vec1::<f32>()?;
        let mut indexed: Vec<(usize, f32)> = logits_t.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let (idx0, l0) = indexed[0];
        let (idx1, l1) = indexed[1];

        let max_l = l0.max(l1);
        let exp0 = (l0 - max_l).exp();
        let exp1 = (l1 - max_l).exp();
        let sum_exp = exp0 + exp1;
        let p0 = exp0 / sum_exp;
        let p1 = exp1 / sum_exp;

        top2_indices_vec.push(idx0 as i64);
        top2_indices_vec.push(idx1 as i64);
        top2_probs_vec.push(p0);
        top2_probs_vec.push(p1);

        let x_t = x.get(t)?; // [D]
        let x_t_row = x_t.reshape((1, d_model))?; // [1, D]

        let w_e0 = w_experts.get(idx0)?;
        let out_e0 = x_t_row.matmul(&w_e0)?.to_vec2::<f32>()?[0].clone();

        let w_e1 = w_experts.get(idx1)?;
        let out_e1 = x_t_row.matmul(&w_e1)?.to_vec2::<f32>()?[0].clone();

        for d in 0..d_model {
            moe_out_vec[t * d_model + d] = p0 * out_e0[d] + p1 * out_e1[d];
        }
    }

    // Build dispatch mask [ep_world_size, num_tokens, 2]
    let mut dispatch_mask_vec = vec![0.0f32; ep_world_size * num_tokens * 2];
    for r in 0..ep_world_size {
        let r_start = (r * experts_per_rank) as i64;
        let r_end = ((r + 1) * experts_per_rank) as i64;
        for t in 0..num_tokens {
            for k in 0..2 {
                let exp = top2_indices_vec[t * 2 + k];
                if exp >= r_start && exp < r_end {
                    dispatch_mask_vec[r * num_tokens * 2 + t * 2 + k] = 1.0;
                }
            }
        }
    }

    let moe_out = Tensor::from_slice(&moe_out_vec, (num_tokens, d_model), device)?;
    let top2_indices = Tensor::from_slice(&top2_indices_vec, (num_tokens, 2), device)?;
    let top2_probs = Tensor::from_slice(&top2_probs_vec, (num_tokens, 2), device)?;
    let dispatch_mask =
        Tensor::from_slice(&dispatch_mask_vec, (ep_world_size, num_tokens, 2), device)?;

    Ok((moe_out, top2_indices, top2_probs, dispatch_mask))
}
