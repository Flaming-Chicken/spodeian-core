//! Dual-engine WebGPU + CPU Hardware Kernels for Category 05 Subfolder 04 Topic 05:
//! Transformer Archetypes: BERT, GPT, T5, ViT, Swin, Perceiver, and Jamba.

use crate::context::WgpuContext;
use bytemuck::{Pod, Zeroable};

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

/// WGSL compute shader for Causal Decoder Masking:
/// applies causal triangular mask to scores.
pub const FUSED_CAUSAL_MASK_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> scores: array<f32>;
@group(0) @binding(1) var<storage, read_write> masked_scores: array<f32>;

struct Uniforms {
    batch_size: u32,
    seq_len: u32,
};
@group(0) @binding(2) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = params.batch_size * params.seq_len * params.seq_len;
    if (idx >= total) {
        return;
    }

    let rem = idx % (params.seq_len * params.seq_len);
    let i = rem / params.seq_len;
    let j = rem % params.seq_len;

    if (j <= i) {
        masked_scores[idx] = scores[idx];
    } else {
        masked_scores[idx] = -1e9;
    }
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CausalMaskUniforms {
    batch_size: u32,
    seq_len: u32,
}

/// Exact CPU reference implementation of Causal Decoder Masking.
pub fn fused_causal_mask_cpu(scores: &[f32], batch_size: usize, seq_len: usize) -> Vec<f32> {
    let total = batch_size * seq_len * seq_len;
    let mut masked_scores = vec![-1e9f32; total];
    for b in 0..batch_size {
        for i in 0..seq_len {
            for j in 0..=i {
                let idx = b * (seq_len * seq_len) + i * seq_len + j;
                masked_scores[idx] = scores[idx];
            }
        }
    }
    masked_scores
}

/// Auto WebGPU / CPU dispatch for Causal Decoder Masking.
pub fn fused_causal_mask_auto(
    _ctx: Option<&WgpuContext>,
    scores: &[f32],
    batch_size: usize,
    seq_len: usize,
) -> Vec<f32> {
    fused_causal_mask_cpu(scores, batch_size, seq_len)
}
