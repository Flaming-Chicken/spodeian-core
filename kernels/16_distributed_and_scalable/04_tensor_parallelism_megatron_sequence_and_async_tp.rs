//! CPU and WGSL kernels for Tensor Parallelism (Megatron-LM, Sequence Parallelism, and Async TP).

/// WGSL compute shader for fused TP row-parallel partial shard AllReduce reduction.
pub const FUSED_TP_ROW_ALLREDUCE_WGSL: &str = r#"
struct Uniforms {
    world_size: u32,
    batch_seq: u32,
    d_model: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> partial_shards: array<f32>; // [world_size, batch_seq, d_model]
@group(0) @binding(2) var<storage, read_write> allreduced_out: array<f32>; // [batch_seq, d_model]

@compute @workgroup_size(16, 16, 1)
fn tp_row_allreduce_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let token_idx = gid.x;
    let feat_idx = gid.y;

    if (token_idx >= u.batch_seq || feat_idx >= u.d_model) { return; }

    var acc: f32 = 0.0;
    let slice_size = u.batch_seq * u.d_model;
    for (var r = 0u; r < u.world_size; r++) {
        acc += partial_shards[r * slice_size + token_idx * u.d_model + feat_idx];
    }

    allreduced_out[token_idx * u.d_model + feat_idx] = acc;
}
"#;

/// CPU linear anchor: `y[b, o] = bias[o] + sum_i x[b, i] * w[i, o]`.
pub fn linear_anchor_cpu(
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    batch: usize,
    d_in: usize,
    d_out: usize,
) -> Vec<f32> {
    let mut y = vec![0.0f32; batch * d_out];
    for (b, row) in y.chunks_mut(d_out).enumerate() {
        for (o, out) in row.iter_mut().enumerate() {
            *out = bias[o]
                + (0..d_in)
                    .map(|i| x[b * d_in + i] * w[i * d_out + o])
                    .sum::<f32>();
        }
    }
    y
}

/// CPU Tensor Parallel row-parallel AllReduce reduction: `out[i] = sum_r partial_shards[r, i]`.
pub fn tp_row_allreduce_cpu(
    partial_shards: &[f32],
    world_size: usize,
    batch_seq: usize,
    d_model: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch_seq * d_model];
    let slice_size = batch_seq * d_model;

    for i in 0..slice_size {
        let mut acc = 0.0f32;
        for r in 0..world_size {
            acc += partial_shards[r * slice_size + i];
        }
        out[i] = acc;
    }
    out
}

/// CPU Sequence Parallelism AllGather simulation: concatenates sequence shards along dim 1.
pub fn sequence_parallel_allgather_cpu(
    seq_shards: &[f32],
    world_size: usize,
    batch: usize,
    seq_per_shard: usize,
    d_model: usize,
) -> Vec<f32> {
    let total_seq = world_size * seq_per_shard;
    let mut full = vec![0.0f32; batch * total_seq * d_model];

    for b in 0..batch {
        for r in 0..world_size {
            let shard_offset = (r * batch * seq_per_shard + b * seq_per_shard) * d_model;
            let full_offset = (b * total_seq + r * seq_per_shard) * d_model;
            let count = seq_per_shard * d_model;
            full[full_offset..full_offset + count]
                .copy_from_slice(&seq_shards[shard_offset..shard_offset + count]);
        }
    }
    full
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tp_row_allreduce_cpu() {
        let world_size = 2;
        let batch_seq = 2;
        let d_model = 3;
        // Rank 0: [1, 2, 3, 4, 5, 6]
        // Rank 1: [2, 3, 4, 5, 6, 7]
        let shards = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let out = tp_row_allreduce_cpu(&shards, world_size, batch_seq, d_model);
        assert_eq!(out, vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0]);
    }

    #[test]
    fn test_sequence_parallel_allgather_cpu() {
        let world_size = 2;
        let batch = 1;
        let seq_per_shard = 2;
        let d_model = 2;
        // Rank 0: 2 tokens of dim 2: [1, 1, 2, 2]
        // Rank 1: 2 tokens of dim 2: [3, 3, 4, 4]
        let seq_shards = vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0];
        let full =
            sequence_parallel_allgather_cpu(&seq_shards, world_size, batch, seq_per_shard, d_model);
        assert_eq!(full, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]);
    }
}
