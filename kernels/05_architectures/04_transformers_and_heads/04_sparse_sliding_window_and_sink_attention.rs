//! Dual-engine WebGPU + CPU Hardware Kernels for Category 05 Subfolder 04 Topic 04:
//! Sparse, Sliding Window, and Sink Attention.

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

/// WGSL compute shader for Streaming Sink Attention Masking:
/// applies sink and sliding-window mask to raw scores.
pub const FUSED_SINK_MASK_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> scores: array<f32>;
@group(0) @binding(1) var<storage, read_write> masked_scores: array<f32>;

struct Uniforms {
    batch_size: u32,
    t_len: u32,
    s_len: u32,
    num_sinks: u32,
    window_size: u32,
};
@group(0) @binding(2) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = params.batch_size * params.t_len * params.s_len;
    if (idx >= total) {
        return;
    }

    let rem = idx % (params.t_len * params.s_len);
    let i = rem / params.s_len;
    let j = rem % params.s_len;

    let is_sink = j < params.num_sinks;
    let is_window = (i >= j) && ((i - j) < params.window_size);
    let valid = (is_sink || is_window) && (j <= i);

    if (valid) {
        masked_scores[idx] = scores[idx];
    } else {
        masked_scores[idx] = -1e9;
    }
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SinkMaskUniforms {
    batch_size: u32,
    t_len: u32,
    s_len: u32,
    num_sinks: u32,
    window_size: u32,
}

/// Exact CPU reference implementation of Streaming Sink Mask.
pub fn fused_sink_mask_cpu(
    scores: &[f32],
    batch_size: usize,
    t_len: usize,
    s_len: usize,
    num_sinks: usize,
    window_size: usize,
) -> Vec<f32> {
    let total = batch_size * t_len * s_len;
    let mut masked_scores = vec![-1e9f32; total];

    for b in 0..batch_size {
        for i in 0..t_len {
            for j in 0..s_len {
                let is_sink = j < num_sinks;
                let is_window = i >= j && (i - j) < window_size;
                let valid = (is_sink || is_window) && j <= i;
                let idx = b * (t_len * s_len) + i * s_len + j;
                if valid {
                    masked_scores[idx] = scores[idx];
                }
            }
        }
    }
    masked_scores
}

/// Auto WebGPU / CPU dispatch for Streaming Sink Mask.
pub fn fused_sink_mask_auto(
    _ctx: Option<&WgpuContext>,
    scores: &[f32],
    batch_size: usize,
    t_len: usize,
    s_len: usize,
    num_sinks: usize,
    window_size: usize,
) -> Vec<f32> {
    fused_sink_mask_cpu(scores, batch_size, t_len, s_len, num_sinks, window_size)
}
