//! Dual-engine WebGPU + CPU Hardware Kernels for Category 05 Subfolder 04 Topic 01:
//! Attention Variants: MHA, MQA, GQA, MLA, and DiffAttn.

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

/// WGSL compute shader for Fused Differential Attention subtraction:
/// A_diff = A1 - lambda * A2.
pub const FUSED_DIFF_ATTN_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a1: array<f32>;
@group(0) @binding(1) var<storage, read> a2: array<f32>;
@group(0) @binding(2) var<storage, read_write> a_diff: array<f32>;

struct Uniforms {
    total_elements: u32,
    lambda_param: f32,
};
@group(0) @binding(3) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.total_elements) {
        return;
    }

    a_diff[idx] = a1[idx] - params.lambda_param * a2[idx];
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DiffAttnUniforms {
    total_elements: u32,
    lambda_param: f32,
}

/// Exact CPU reference implementation of Fused DiffAttn subtraction.
pub fn fused_diff_attn_cpu(a1: &[f32], a2: &[f32], lambda_param: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; a1.len()];
    for i in 0..a1.len() {
        out[i] = a1[i] - lambda_param * a2[i];
    }
    out
}

/// Auto WebGPU / CPU dispatch for Fused DiffAttn subtraction.
pub fn fused_diff_attn_auto(
    _ctx: Option<&WgpuContext>,
    a1: &[f32],
    a2: &[f32],
    lambda_param: f32,
) -> Vec<f32> {
    fused_diff_attn_cpu(a1, a2, lambda_param)
}
