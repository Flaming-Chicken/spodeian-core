//! DiLoCo Local SGD, PowerSGD, and Distributed Checkpointing in candle/16_distributed_and_scalable.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// DiLoCo outer optimization step across worker clusters.
pub fn diloco_outer_step(
    theta_0: &Tensor,
    theta_workers: &Tensor,
    v_outer: &Tensor,
    lr_outer: f64,
    beta_outer: f64,
) -> Result<(Tensor, Tensor, Tensor)> {
    let w = theta_workers.dim(0)? as f64;
    let theta_0_expanded = theta_0.unsqueeze(0)?;
    let deltas = theta_0_expanded.broadcast_sub(theta_workers)?;
    let delta_avg = (deltas.sum(0)? / w)?;
    let v_scaled = (v_outer * beta_outer)?;
    let v_new = (v_scaled + &delta_avg)?;
    let step = (&v_new * lr_outer)?;
    let theta_new = (theta_0 - step)?;
    Ok((theta_new, v_new, delta_avg))
}

/// PowerSGD rank-r gradient compression and reconstruction.
pub fn powersgd_compress_decompress(
    grad_matrix: &Tensor,
    probe_q: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    let p = grad_matrix.matmul(probe_q)?;
    let sum_sq = p.sqr()?.sum_all()?.to_scalar::<f32>()?;
    let norm = (sum_sq as f64).sqrt() + 1e-8;
    let p_norm = (p / norm)?;

    let grad_t = grad_matrix.t()?;
    let q_new = grad_t.matmul(&p_norm)?;
    let q_new_t = q_new.t()?;
    let recon = p_norm.matmul(&q_new_t)?;
    Ok((recon, p_norm, q_new))
}

/// Distributed checkpoint resharding from 2 shards to 4 shards.
pub fn distributed_checkpoint_reshard(shards_2: &Tensor) -> Result<(Tensor, Tensor)> {
    let s0 = shards_2.get(0)?;
    let s1 = shards_2.get(1)?;
    let full_tensor = Tensor::cat(&[&s0, &s1], 0)?;

    let m = full_tensor.dim(0)?;
    let chunk_size = m / 4;
    let mut chunks = Vec::with_capacity(4);
    for i in 0..4 {
        chunks.push(full_tensor.narrow(0, i * chunk_size, chunk_size)?);
    }
    let refs: Vec<&Tensor> = chunks.iter().collect();
    let shards_4 = Tensor::stack(&refs, 0)?;
    Ok((full_tensor, shards_4))
}
