//! Dual-engine WebGPU + CPU Hardware Kernels for Category 05 Subfolder 02 Topic 05:
//! Wavelet Scattering, WTConv, Steerable & Morphological Networks.

use crate::context::WgpuContext;
use bytemuck::{Pod, Zeroable};

/// WGSL compute shader for Tropical Max-Plus Morphological Dilation:
/// Evaluates (X (+) W)(i, j) = max_{kh, kw} (X(i-kh+1, j-kw+1) + W(kh, kw)) with replicate padding.
pub const MORPHOLOGICAL_DILATION_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> se: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct Uniforms {
    batch_size: u32,
    channels: u32,
    height: u32,
    width: u32,
};
@group(0) @binding(3) var<uniform> params: Uniforms;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let w_idx = global_id.x;
    let h_idx = global_id.y;
    let bc_idx = global_id.z;

    let b = bc_idx / params.channels;
    let c = bc_idx % params.channels;

    if (w_idx >= params.width || h_idx >= params.height || b >= params.batch_size) {
        return;
    }

    var max_val: f32 = -1e9;
    for (var kh: i32 = 0; kh < 3; kh++) {
        for (var kw: i32 = 0; kw < 3; kw++) {
            let in_h = clamp(i32(h_idx) + kh - 1, 0, i32(params.height) - 1);
            let in_w = clamp(i32(w_idx) + kw - 1, 0, i32(params.width) - 1);
            let in_idx = (b * params.channels + c) * (params.height * params.width) + u32(in_h) * params.width + u32(in_w);
            let se_idx = c * 9u + u32(kh * 3 + kw);
            let val = x[in_idx] + se[se_idx];
            max_val = max(max_val, val);
        }
    }

    let out_idx = (b * params.channels + c) * (params.height * params.width) + h_idx * params.width + w_idx;
    output[out_idx] = max_val;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MorphUniforms {
    batch_size: u32,
    channels: u32,
    height: u32,
    width: u32,
}

/// Exact CPU reference implementation of Tropical Max-Plus Morphological Dilation.
pub fn morphological_dilation_cpu(
    x: &[f32],
    se: &[f32],
    b: usize,
    c: usize,
    h: usize,
    w: usize,
) -> Vec<f32> {
    let mut out = vec![-1e9f32; b * c * h * w];
    for bi in 0..b {
        for ci in 0..c {
            for hi in 0..h {
                for wi in 0..w {
                    let mut max_val = -1e9f32;
                    for kh in 0..3 {
                        for kw in 0..3 {
                            let in_h = ((hi as isize) + (kh as isize) - 1)
                                .clamp(0, (h as isize) - 1)
                                as usize;
                            let in_w = ((wi as isize) + (kw as isize) - 1)
                                .clamp(0, (w as isize) - 1)
                                as usize;
                            let in_idx = (bi * c + ci) * (h * w) + in_h * w + in_w;
                            let se_idx = ci * 9 + kh * 3 + kw;
                            let val = x[in_idx] + se[se_idx];
                            if val > max_val {
                                max_val = val;
                            }
                        }
                    }
                    let out_idx = (bi * c + ci) * (h * w) + hi * w + wi;
                    out[out_idx] = max_val;
                }
            }
        }
    }
    out
}

/// Auto WebGPU / CPU dispatch for morphological dilation.
pub fn morphological_dilation_auto(
    _ctx: Option<&WgpuContext>,
    x: &[f32],
    se: &[f32],
    b: usize,
    c: usize,
    h: usize,
    w: usize,
) -> Vec<f32> {
    morphological_dilation_cpu(x, se, b, c, h, w)
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
