//! Context Parallelism Ring Attention and Ulysses in candle/16_distributed_and_scalable.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Standard scaled dot-product attention in Candle: [B, H, S, D].
pub fn standard_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let d_k = q.dim(3)? as f64;
    let scale = 1.0 / d_k.sqrt();
    let k_t = k.transpose(2, 3)?;
    let scores = (q.matmul(&k_t)? * scale)?;
    let weights = candle_nn::ops::softmax_last_dim(&scores)?;
    weights.matmul(v)
}

/// Ring Attention forward pass across CP world_size ranks.
pub fn ring_attention_forward(
    q_shards: &Tensor,
    k_shards: &Tensor,
    v_shards: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let world_size = q_shards.dim(0)?;
    let d_k = q_shards.dim(4)? as f64;
    let scale = 1.0 / d_k.sqrt();

    let mut k_unstacked = Vec::with_capacity(world_size);
    let mut v_unstacked = Vec::with_capacity(world_size);
    for i in 0..world_size {
        k_unstacked.push(k_shards.get(i)?);
        v_unstacked.push(v_shards.get(i)?);
    }
    let k_refs: Vec<&Tensor> = k_unstacked.iter().collect();
    let v_refs: Vec<&Tensor> = v_unstacked.iter().collect();
    let k_full = Tensor::cat(&k_refs, 2)?;
    let v_full = Tensor::cat(&v_refs, 2)?;
    let k_full_t = k_full.transpose(2, 3)?;

    let mut out_shards_list = Vec::with_capacity(world_size);
    for r in 0..world_size {
        let q_r = q_shards.get(r)?;
        let scores_r = (q_r.matmul(&k_full_t)? * scale)?;
        let weights_r = candle_nn::ops::softmax_last_dim(&scores_r)?;
        let out_r = weights_r.matmul(&v_full)?;
        out_shards_list.push(out_r);
    }

    let out_refs: Vec<&Tensor> = out_shards_list.iter().collect();
    let out_shards = Tensor::stack(&out_refs, 0)?;
    let out_full = Tensor::cat(&out_refs, 2)?;
    Ok((out_shards, out_full))
}

/// DeepSpeed Ulysses All-to-All context parallel attention.
pub fn deepspeed_ulysses_forward(
    q_seq_shards: &Tensor,
    k_seq_shards: &Tensor,
    v_seq_shards: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let world_size = q_seq_shards.dim(0)?;
    let h = q_seq_shards.dim(2)?;
    let s_chunk = q_seq_shards.dim(3)?;
    let h_per_rank = h / world_size;

    // All-to-All 1: sequence-sharded to head-sharded
    let mut q_head_list = Vec::with_capacity(world_size);
    let mut k_head_list = Vec::with_capacity(world_size);
    let mut v_head_list = Vec::with_capacity(world_size);

    for r in 0..world_size {
        let mut q_slices = Vec::with_capacity(world_size);
        let mut k_slices = Vec::with_capacity(world_size);
        let mut v_slices = Vec::with_capacity(world_size);
        for src in 0..world_size {
            q_slices.push(
                q_seq_shards
                    .get(src)?
                    .narrow(1, r * h_per_rank, h_per_rank)?,
            );
            k_slices.push(
                k_seq_shards
                    .get(src)?
                    .narrow(1, r * h_per_rank, h_per_rank)?,
            );
            v_slices.push(
                v_seq_shards
                    .get(src)?
                    .narrow(1, r * h_per_rank, h_per_rank)?,
            );
        }
        let q_refs: Vec<&Tensor> = q_slices.iter().collect();
        let k_refs: Vec<&Tensor> = k_slices.iter().collect();
        let v_refs: Vec<&Tensor> = v_slices.iter().collect();
        q_head_list.push(Tensor::cat(&q_refs, 2)?);
        k_head_list.push(Tensor::cat(&k_refs, 2)?);
        v_head_list.push(Tensor::cat(&v_refs, 2)?);
    }

    // Local attention on full sequence S for subset of heads
    let mut out_head_list = Vec::with_capacity(world_size);
    for r in 0..world_size {
        let out_r = standard_attention(&q_head_list[r], &k_head_list[r], &v_head_list[r])?;
        out_head_list.push(out_r);
    }

    // All-to-All 2: head-sharded back to sequence-sharded
    let mut out_seq_list = Vec::with_capacity(world_size);
    for r in 0..world_size {
        let mut slices = Vec::with_capacity(world_size);
        for src in 0..world_size {
            slices.push(out_head_list[src].narrow(2, r * s_chunk, s_chunk)?);
        }
        let refs: Vec<&Tensor> = slices.iter().collect();
        out_seq_list.push(Tensor::cat(&refs, 1)?);
    }

    let out_refs: Vec<&Tensor> = out_seq_list.iter().collect();
    let out_seq_shards = Tensor::stack(&out_refs, 0)?;
    let out_full = Tensor::cat(&out_refs, 2)?;
    Ok((out_seq_shards, out_full))
}
