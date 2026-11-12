//! CPU and WGSL kernels for Supervised Finetuning Chat Masks Neftune and Sequence Packing.

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

/// Sequence packing block-diagonal causal attention mask CPU kernel:
/// mask[i, j] = 1.0 if j <= i and seg[i] == seg[j] else 0.0.
pub fn packed_block_diagonal_causal_mask_cpu(segment_ids: &[i64], seq_len: usize) -> Vec<f32> {
    let mut mask = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        let offset = i * seq_len;
        for j in 0..=i {
            if segment_ids[i] == segment_ids[j] {
                mask[offset + j] = 1.0f32;
            }
        }
    }
    mask
}

/// Sequence packing block-diagonal causal attention mask WGSL compute shader
pub const PACKED_BLOCK_DIAGONAL_CAUSAL_MASK_WGSL: &str = r#"
struct Config {
    seq_len: u32,
};

@group(0) @binding(0) var<storage, read> segment_ids: array<u32>;
@group(0) @binding(1) var<storage, read_write> mask: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Config;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.y;
    let j = global_id.x;

    if (i >= cfg.seq_len || j >= cfg.seq_len) {
        return;
    }

    let idx = i * cfg.seq_len + j;
    if (j <= i && segment_ids[i] == segment_ids[j]) {
        mask[idx] = 1.0;
    } else {
        mask[idx] = 0.0;
    }
}
"#;
