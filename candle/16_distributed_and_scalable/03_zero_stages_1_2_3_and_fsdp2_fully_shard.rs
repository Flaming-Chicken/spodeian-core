//! Zero Stages 1 2 3 and FSDP2 Fully Shard
//! in candle/16_distributed_and_scalable.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// FSDP2 AllGather: reconstructs full 2D weight matrix from parameter shards.
pub fn fsdp2_allgather_param(param_shards: &Tensor) -> Result<Tensor> {
    let world_size = param_shards.dim(0)?;
    let mut unstacked = Vec::with_capacity(world_size);
    for r in 0..world_size {
        unstacked.push(param_shards.get(r)?);
    }
    let refs: Vec<&Tensor> = unstacked.iter().collect();
    Tensor::cat(&refs, 0)
}

/// FSDP2 forward: AllGather parameter shards, compute matmul.
pub fn fsdp2_forward_matmul(x: &Tensor, param_shards: &Tensor) -> Result<(Tensor, Tensor)> {
    let w_full = fsdp2_allgather_param(param_shards)?;
    let h = x.matmul(&w_full)?;
    Ok((h, w_full))
}

/// FSDP2 backward: compute full gradient and ReduceScatter across ranks.
pub fn fsdp2_backward_reducescatter(
    x: &Tensor,
    grad_out: &Tensor,
    world_size: usize,
) -> Result<(Tensor, Tensor)> {
    let grad_w_full = x.t()?.matmul(grad_out)?;
    let d_in = grad_w_full.dim(0)?;
    let chunk_size = d_in / world_size;

    let mut shards = Vec::with_capacity(world_size);
    for r in 0..world_size {
        shards.push(grad_w_full.narrow(0, r * chunk_size, chunk_size)?);
    }
    let refs: Vec<&Tensor> = shards.iter().collect();
    let grad_w_shards = Tensor::stack(&refs, 0)?;
    Ok((grad_w_full, grad_w_shards))
}

/// ZeRO-1 Adam optimizer step on partitioned parameter slice.
pub fn zero1_adam_step(
    p: &Tensor,
    g: &Tensor,
    m: &Tensor,
    v: &Tensor,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    step: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let m_new = ((m * beta1)? + (g * (1.0 - beta1))?)?;
    let g_sq = g.mul(g)?;
    let v_new = ((v * beta2)? + (&g_sq * (1.0 - beta2))?)?;

    let bias_corr1 = 1.0 - beta1.powi(step as i32);
    let bias_corr2 = 1.0 - beta2.powi(step as i32);

    let m_hat = (&m_new / bias_corr1)?;
    let v_hat = (&v_new / bias_corr2)?;

    let denom = v_hat.sqrt()?.affine(1.0, eps)?;
    let step_val = (m_hat.div(&denom)? * lr)?;
    let p_new = p.sub(&step_val)?;

    Ok((p_new, m_new, v_new))
}
