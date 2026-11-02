//! Collectives and DDP Gradient Bucketing Mechanics
//! in candle/16_distributed_and_scalable.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Element-wise AllReduce sum across simulated ranks: sum(rank_data, dim=0).
pub fn ring_allreduce_simulate(rank_data: &Tensor) -> Result<Tensor> {
    rank_data.sum(0)
}

/// DDP gradient bucketing with asynchronous bucket-level reduction simulation.
pub fn ddp_bucketed_allreduce(
    rank_grads: &Tensor,
    bucket_cap: usize,
) -> Result<(Tensor, Vec<Tensor>)> {
    let dims = rank_grads.dims();
    if dims.len() != 2 {
        candle_core::bail!("rank_grads must be a 2D tensor [world_size, num_params]");
    }
    let world_size = dims[0];
    let num_params = dims[1];
    let num_buckets = (num_params + bucket_cap - 1) / bucket_cap;

    let mut bucket_results = Vec::with_capacity(num_buckets);
    for b in 0..num_buckets {
        let start = b * bucket_cap;
        let end = (start + bucket_cap).min(num_params);
        let len = end - start;

        let bucket_slice = rank_grads.narrow(1, start, len)?;
        let reduced = (bucket_slice.sum(0)? / (world_size as f64))?;
        bucket_results.push(reduced);
    }

    let tensor_refs: Vec<&Tensor> = bucket_results.iter().collect();
    let averaged_grads = Tensor::cat(&tensor_refs, 0)?;
    Ok((averaged_grads, bucket_results))
}
