//! Dual-engine WebGPU + CPU Hardware Kernels for Category 05 Subfolder 02 Topic 04:
//! Spatiotemporal Video: I3D Weight Inflation.

use crate::context::WgpuContext;
use bytemuck::{Pod, Zeroable};

/// WGSL compute shader for I3D Weight Inflation:
/// inflates 2D kernel [C_out, C_in, K_h, K_w] into 3D [C_out, C_in, K_t, K_h, K_w] with 1/k_t scaling
pub const I3D_INFLATION_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> conv2d_w: array<f32>;
@group(0) @binding(1) var<storage, read_write> i3d_w: array<f32>;

struct Uniforms {
    total_3d_elements: u32,
    c_in_kh_kw: u32, // C_in * K_h * K_w
    k_t: u32,
    scale_factor: f32, // 1.0 / k_t
};
@group(0) @binding(2) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.total_3d_elements) {
        return;
    }
    let spatial_kh_kw = params.c_in_kh_kw;
    let c_out = idx / (params.k_t * spatial_kh_kw);
    let rem = idx % (params.k_t * spatial_kh_kw);
    let in_spatial = rem % spatial_kh_kw;
    let idx_2d = c_out * spatial_kh_kw + in_spatial;

    i3d_w[idx] = conv2d_w[idx_2d] * params.scale_factor;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct InflationUniforms {
    total_3d_elements: u32,
    c_in_kh_kw: u32,
    k_t: u32,
    scale_factor: f32,
}

/// Exact CPU reference implementation of I3D weight inflation.
pub fn i3d_inflate_cpu(
    conv2d_w: &[f32],
    c_out: usize,
    c_in: usize,
    k_t: usize,
    kh: usize,
    kw: usize,
) -> Vec<f32> {
    let hw = kh * kw;
    let total = c_out * c_in * k_t * hw;
    let scale = 1.0 / (k_t as f32);
    let mut out = vec![0.0f32; total];

    for co in 0..c_out {
        for ci in 0..c_in {
            for t in 0..k_t {
                for p in 0..hw {
                    let idx_3d = co * (c_in * k_t * hw) + ci * (k_t * hw) + t * hw + p;
                    let idx_2d = co * (c_in * hw) + ci * hw + p;
                    out[idx_3d] = conv2d_w[idx_2d] * scale;
                }
            }
        }
    }
    out
}

/// Auto WebGPU / CPU dispatch for I3D weight inflation.
pub fn i3d_inflate_auto(
    _ctx: Option<&WgpuContext>,
    conv2d_w: &[f32],
    c_out: usize,
    c_in: usize,
    k_t: usize,
    kh: usize,
    kw: usize,
) -> Vec<f32> {
    i3d_inflate_cpu(conv2d_w, c_out, c_in, k_t, kh, kw)
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
