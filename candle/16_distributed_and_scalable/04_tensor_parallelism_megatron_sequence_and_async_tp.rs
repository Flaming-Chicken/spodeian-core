//! Tensor Parallelism Megatron Sequence and Async TP
//! in candle/16_distributed_and_scalable.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Helper for 3D tensor multiplied by 2D weight matrix: [B, S, D] @ [D, Out] -> [B, S, Out].
fn matmul_3d_2d(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    if dims.len() == 3 {
        let (b, s, d) = (dims[0], dims[1], dims[2]);
        let x_2d = x.reshape((b * s, d))?;
        let out_2d = x_2d.matmul(w)?;
        let out_d = out_2d.dim(1)?;
        out_2d.reshape((b, s, out_d))
    } else {
        x.matmul(w)
    }
}

/// Megatron-LM 1D Tensor Parallel MLP block.
pub fn megatron_tp_mlp(
    x: &Tensor,
    w1_shards: &Tensor,
    w2_shards: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let world_size = w1_shards.dim(0)?;
    let mut partials = Vec::with_capacity(world_size);

    for r in 0..world_size {
        let w1_r = w1_shards.get(r)?;
        let w2_r = w2_shards.get(r)?;

        let proj1 = matmul_3d_2d(x, &w1_r)?;
        let h_r = proj1.gelu_erf()?;
        let y_r = matmul_3d_2d(&h_r, &w2_r)?;
        partials.push(y_r);
    }

    let refs: Vec<&Tensor> = partials.iter().collect();
    let partial_shards = Tensor::stack(&refs, 0)?;
    let y = partial_shards.sum(0)?;
    Ok((y, partial_shards))
}

/// Megatron Sequence Parallelism: AllGather, TP MLP, ReduceScatter.
pub fn sequence_parallel_mlp(
    x_seq_shards: &Tensor,
    w1_shards: &Tensor,
    w2_shards: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let world_size = x_seq_shards.dim(0)?;

    let mut unstacked = Vec::with_capacity(world_size);
    for r in 0..world_size {
        unstacked.push(x_seq_shards.get(r)?);
    }
    let refs: Vec<&Tensor> = unstacked.iter().collect();
    let x_full = Tensor::cat(&refs, 1)?;

    let (y_full, _) = megatron_tp_mlp(&x_full, w1_shards, w2_shards)?;

    let seq_len = y_full.dim(1)?;
    let chunk_size = seq_len / world_size;
    let mut chunks = Vec::with_capacity(world_size);
    for r in 0..world_size {
        chunks.push(y_full.narrow(1, r * chunk_size, chunk_size)?);
    }
    let chunk_refs: Vec<&Tensor> = chunks.iter().collect();
    let y_seq_shards = Tensor::stack(&chunk_refs, 0)?;

    Ok((y_seq_shards, x_full))
}
