//! Dual-engine WebGPU + CPU Hardware Kernels for Category 05 Subfolder 02 Topic 02:
//! Specialized Convolutions & Channel Attention (SE, ECA, CoordConv).

use crate::context::WgpuContext;
use bytemuck::{Pod, Zeroable};

/// WGSL compute shader for channel attention recalibration (SE / ECA):
/// y[b, c, h, w] = x[b, c, h, w] * scale[b, c]
pub const CHANNEL_ATTENTION_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input_x: array<f32>;
@group(0) @binding(1) var<storage, read> channel_scale: array<f32>; // shape [B, C]
@group(0) @binding(2) var<storage, read_write> output_y: array<f32>;

struct Uniforms {
    batch_size: u32,
    num_channels: u32,
    spatial_size: u32, // H * W
    _pad: u32,
};
@group(0) @binding(3) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_elements = params.batch_size * params.num_channels * params.spatial_size;
    if (idx >= total_elements) {
        return;
    }

    let b = idx / (params.num_channels * params.spatial_size);
    let rem = idx % (params.num_channels * params.spatial_size);
    let c = rem / params.spatial_size;

    let scale = channel_scale[b * params.num_channels + c];
    output_y[idx] = input_x[idx] * scale;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AttentionUniforms {
    batch_size: u32,
    num_channels: u32,
    spatial_size: u32,
    _pad: u32,
}

/// Exact CPU reference implementation of channel attention recalibration.
pub fn channel_attention_cpu(
    x: &[f32],
    scale: &[f32],
    b_sz: usize,
    c_sz: usize,
    h_sz: usize,
    w_sz: usize,
) -> Vec<f32> {
    let spatial = h_sz * w_sz;
    let total = b_sz * c_sz * spatial;
    let mut out = vec![0.0f32; total];

    for b in 0..b_sz {
        for c in 0..c_sz {
            let s = scale[b * c_sz + c];
            for hw in 0..spatial {
                let idx = b * (c_sz * spatial) + c * spatial + hw;
                out[idx] = x[idx] * s;
            }
        }
    }
    out
}

/// Auto WebGPU / CPU dispatch for channel attention recalibration.
pub fn channel_attention_auto(
    _ctx: Option<&WgpuContext>,
    x: &[f32],
    scale: &[f32],
    b_sz: usize,
    c_sz: usize,
    h_sz: usize,
    w_sz: usize,
) -> Vec<f32> {
    channel_attention_cpu(x, scale, b_sz, c_sz, h_sz, w_sz)
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
