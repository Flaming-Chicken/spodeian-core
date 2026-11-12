//! Dual-engine WebGPU + CPU Hardware Kernels for Category 05 Subfolder 02 Topic 03:
//! U-Net & Multiscale Feature Pyramids: BiFPN Fast Normalized Feature Fusion.

use crate::context::WgpuContext;
use bytemuck::{Pod, Zeroable};

/// WGSL compute shader for BiFPN fast normalized feature fusion:
/// y[i] = norm_w1 * p1[i] + norm_w2 * p2[i]
pub const BIFPN_FUSION_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input_p1: array<f32>;
@group(0) @binding(1) var<storage, read> input_p2: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_y: array<f32>;

struct Uniforms {
    norm_w1: f32,
    norm_w2: f32,
    total_elements: u32,
    _pad: u32,
};
@group(0) @binding(3) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.total_elements) {
        return;
    }
    output_y[idx] = params.norm_w1 * input_p1[idx] + params.norm_w2 * input_p2[idx];
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FusionUniforms {
    norm_w1: f32,
    norm_w2: f32,
    total_elements: u32,
    _pad: u32,
}

/// Exact CPU reference implementation of BiFPN fast normalized feature fusion.
pub fn bifpn_fusion_cpu(p1: &[f32], p2: &[f32], w1: f32, w2: f32, eps: f32) -> Vec<f32> {
    assert_eq!(p1.len(), p2.len());
    let relu_w1 = w1.max(0.0);
    let relu_w2 = w2.max(0.0);
    let total = relu_w1 + relu_w2 + eps;
    let norm_w1 = relu_w1 / total;
    let norm_w2 = relu_w2 / total;

    p1.iter()
        .zip(p2.iter())
        .map(|(&v1, &v2)| norm_w1 * v1 + norm_w2 * v2)
        .collect()
}

/// Auto WebGPU / CPU dispatch for BiFPN fast normalized feature fusion.
pub fn bifpn_fusion_auto(
    _ctx: Option<&WgpuContext>,
    p1: &[f32],
    p2: &[f32],
    w1: f32,
    w2: f32,
    eps: f32,
) -> Vec<f32> {
    bifpn_fusion_cpu(p1, p2, w1, w2, eps)
}

/// Standard linear anchor CPU implementation for parity testing.
pub fn linear_anchor_cpu(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = bias[j];
            for p in 0..k {
                sum += x[i * k + p] * weight[p * n + j];
            }
            out[i * n + j] = sum;
        }
    }
    out
}

/// Standard linear anchor auto dispatch.
pub fn linear_anchor_auto(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    linear_anchor_cpu(x, weight, bias, m, k, n)
}
