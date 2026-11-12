//! CPU and WGSL kernels for Context Parallelism (Ring Attention and DeepSpeed Ulysses).

/// WGSL compute shader for fused Ring Attention block score computation: `S[i, j] = sum_k (Q[i, k] * K[j, k]) * scale`.
pub const FUSED_RING_ATTN_SCORES_WGSL: &str = r#"
struct Uniforms {
    seq_chunk_q: u32,
    seq_chunk_k: u32,
    head_dim: u32,
    scale: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> q_chunk: array<f32>; // [seq_chunk_q, head_dim]
@group(0) @binding(2) var<storage, read> k_chunk: array<f32>; // [seq_chunk_k, head_dim]
@group(0) @binding(3) var<storage, read_write> scores_out: array<f32>; // [seq_chunk_q, seq_chunk_k]

@compute @workgroup_size(16, 16, 1)
fn ring_attn_scores_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let col = gid.y;

    if (row >= u.seq_chunk_q || col >= u.seq_chunk_k) { return; }

    var acc: f32 = 0.0;
    for (var k = 0u; k < u.head_dim; k++) {
        acc += q_chunk[row * u.head_dim + k] * k_chunk[col * u.head_dim + k];
    }

    scores_out[row * u.seq_chunk_k + col] = acc * u.scale;
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

/// CPU Ring Attention block score matrix calculation: `scores[q, k] = sum_d Q[q, d] * K[k, d] * scale`.
pub fn ring_attention_block_logits_cpu(
    q: &[f32],
    k: &[f32],
    seq_q: usize,
    seq_k: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    let mut scores = vec![0.0f32; seq_q * seq_k];
    for i in 0..seq_q {
        for j in 0..seq_k {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[i * head_dim + d] * k[j * head_dim + d];
            }
            scores[i * seq_k + j] = dot * scale;
        }
    }
    scores
}

/// CPU All-to-All transposition simulation for DeepSpeed Ulysses:
/// converts sequence-partitioned heads to head-partitioned full sequences.
pub fn ulysses_seq_to_head_alltoall_cpu(
    seq_sharded: &[f32],
    world_size: usize,
    s_chunk: usize,
    num_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let h_per_rank = num_heads / world_size;
    let total_seq = world_size * s_chunk;
    let mut head_sharded = vec![0.0f32; world_size * total_seq * h_per_rank * head_dim];

    for r in 0..world_size {
        for src in 0..world_size {
            for s in 0..s_chunk {
                for h in 0..h_per_rank {
                    let global_head = r * h_per_rank + h;
                    let src_idx = ((src * s_chunk + s) * num_heads + global_head) * head_dim;
                    let dst_token = src * s_chunk + s;
                    let dst_idx = ((r * total_seq + dst_token) * h_per_rank + h) * head_dim;
                    for d in 0..head_dim {
                        head_sharded[dst_idx + d] = seq_sharded[src_idx + d];
                    }
                }
            }
        }
    }
    head_sharded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_attention_block_logits_cpu() {
        let q = vec![1.0, 0.0, 0.0, 1.0]; // 2 tokens of dim 2
        let k = vec![1.0, 0.0, 0.0, 1.0]; // 2 tokens of dim 2
        let scores = ring_attention_block_logits_cpu(&q, &k, 2, 2, 2, 0.5);
        assert_eq!(scores, vec![0.5, 0.0, 0.0, 0.5]);
    }

    #[test]
    fn test_ulysses_seq_to_head_alltoall_cpu() {
        let world_size = 2;
        let s_chunk = 1;
        let num_heads = 2;
        let head_dim = 1;
        // Total seq = 2, h_per_rank = 1.
        // Rank 0 (seq token 0): heads [10.0, 20.0]
        // Rank 1 (seq token 1): heads [30.0, 40.0]
        let seq_sharded = vec![10.0, 20.0, 30.0, 40.0];
        let head_sharded = ulysses_seq_to_head_alltoall_cpu(
            &seq_sharded,
            world_size,
            s_chunk,
            num_heads,
            head_dim,
        );
        // Rank 0 gets head 0: token 0 (10.0), token 1 (30.0) -> [10.0, 30.0]
        // Rank 1 gets head 1: token 0 (20.0), token 1 (40.0) -> [20.0, 40.0]
        assert_eq!(head_sharded, vec![10.0, 30.0, 20.0, 40.0]);
    }
}
