//! CPU and WGSL kernels for KV Cache Paged Radix H2O Snapkv and Yoco.

/// WGSL compute shader for PagedAttention key-query dot product and scoring across physical blocks.
pub const PAGED_ATTENTION_SCORE_WGSL: &str = r#"
struct Uniforms {
    batch: u32,
    context_len: u32,
    block_size: u32,
    dim: u32,
    max_blocks_per_seq: u32,
    scale: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> query: array<f32>; // [batch, 1, dim]
@group(0) @binding(2) var<storage, read> key_blocks: array<f32>; // [num_blocks, block_size, dim]
@group(0) @binding(3) var<storage, read> block_table: array<f32>; // [batch, max_blocks]
@group(0) @binding(4) var<storage, read_write> scores: array<f32>; // [batch, context_len]

@compute @workgroup_size(64, 1, 1)
fn paged_score_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x; // logical token index
    let b = gid.y; // batch index

    if (b >= u.batch || t >= u.context_len) { return; }

    let logical_block = t / u.block_size;
    let block_offset = t % u.block_size;
    let phys_block = u32(block_table[b * u.max_blocks_per_seq + logical_block]);

    var dot: f32 = 0.0;
    let q_start = b * u.dim;
    let k_start = (phys_block * u.block_size + block_offset) * u.dim;

    for (var d = 0u; d < u.dim; d++) {
        dot += query[q_start + d] * key_blocks[k_start + d];
    }

    scores[b * u.context_len + t] = dot * u.scale;
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

/// CPU kernel for gathering paged tokens into contiguous buffer.
pub fn paged_kv_gather_cpu(
    blocks: &[f32],
    block_table: &[f32],
    batch: usize,
    max_blocks: usize,
    context_len: usize,
    block_size: usize,
    dim: usize,
) -> Vec<f32> {
    let mut gathered = vec![0.0f32; batch * context_len * dim];
    for b in 0..batch {
        for t in 0..context_len {
            let logical_block = t / block_size;
            let offset = t % block_size;
            let phys_block = block_table[b * max_blocks + logical_block] as usize;

            let src = (phys_block * block_size + offset) * dim;
            let dst = (b * context_len + t) * dim;
            gathered[dst..dst + dim].copy_from_slice(&blocks[src..src + dim]);
        }
    }
    gathered
}

/// CPU kernel for H2O retention mask computation.
pub fn h2o_evict_mask_cpu(
    scores: &[f32],
    batch: usize,
    seq_len: usize,
    local_window: usize,
    heavy_budget: usize,
) -> Vec<f32> {
    let mut mask = vec![0.0f32; batch * seq_len];
    let recent_start = seq_len.saturating_sub(local_window);

    for b in 0..batch {
        let b_offset = b * seq_len;
        for t in recent_start..seq_len {
            mask[b_offset + t] = 1.0;
        }

        let cand_len = recent_start;
        if cand_len > 0 && heavy_budget > 0 {
            let mut indexed: Vec<(usize, f32)> =
                (0..cand_len).map(|t| (t, scores[b_offset + t])).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            for &(idx, _) in indexed.iter().take(heavy_budget) {
                mask[b_offset + idx] = 1.0;
            }
        }
    }
    mask
}
