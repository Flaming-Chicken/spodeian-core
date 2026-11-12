//! Device Mesh and DTensor Sharding Placements
//! in candle/16_distributed_and_scalable.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Shard a global tensor along shard_dim across mesh_size virtual ranks: Shard(dim).
pub fn dtensor_shard(tensor: &Tensor, mesh_size: usize, shard_dim: usize) -> Result<Tensor> {
    let dim_len = tensor.dim(shard_dim)?;
    if dim_len % mesh_size != 0 {
        candle_core::bail!(
            "tensor dim {} is not divisible by mesh_size {}",
            dim_len,
            mesh_size
        );
    }
    let chunk_size = dim_len / mesh_size;

    let mut shards = Vec::with_capacity(mesh_size);
    for r in 0..mesh_size {
        let chunk = tensor.narrow(shard_dim, r * chunk_size, chunk_size)?;
        shards.push(chunk);
    }

    let refs: Vec<&Tensor> = shards.iter().collect();
    Tensor::stack(&refs, 0)
}

/// AllGather local shards along gather_dim: Shard(dim) -> Replicate().
pub fn dtensor_allgather(shards: &Tensor, gather_dim: usize) -> Result<Tensor> {
    let mesh_size = shards.dim(0)?;
    let mut unstacked = Vec::with_capacity(mesh_size);
    for r in 0..mesh_size {
        unstacked.push(shards.get(r)?);
    }

    let refs: Vec<&Tensor> = unstacked.iter().collect();
    Tensor::cat(&refs, gather_dim)
}

/// Column-parallel matmul: x is Replicated, W is Sharded on columns (dim 1).
pub fn dtensor_column_parallel_matmul(x: &Tensor, w_shards: &Tensor) -> Result<Tensor> {
    let mesh_size = w_shards.dim(0)?;
    let mut outs = Vec::with_capacity(mesh_size);
    for r in 0..mesh_size {
        let w_r = w_shards.get(r)?;
        let out_r = x.matmul(&w_r)?;
        outs.push(out_r);
    }

    let refs: Vec<&Tensor> = outs.iter().collect();
    Tensor::stack(&refs, 0)
}

/// Row-parallel matmul: x is Sharded on cols (dim 1), W is Sharded on rows (dim 0).
pub fn dtensor_row_parallel_matmul(
    x_shards: &Tensor,
    w_shards: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let mesh_size = x_shards.dim(0)?;
    let mut partials = Vec::with_capacity(mesh_size);
    for r in 0..mesh_size {
        let x_r = x_shards.get(r)?;
        let w_r = w_shards.get(r)?;
        let p_r = x_r.matmul(&w_r)?;
        partials.push(p_r);
    }

    let refs: Vec<&Tensor> = partials.iter().collect();
    let partial_shards = Tensor::stack(&refs, 0)?;
    let allreduced = partial_shards.sum(0)?;
    Ok((allreduced, partial_shards))
}
