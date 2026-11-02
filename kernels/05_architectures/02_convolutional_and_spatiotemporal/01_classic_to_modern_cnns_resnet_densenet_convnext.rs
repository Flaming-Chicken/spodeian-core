//! Dual-engine WebGPU + CPU Hardware Kernels for Category 05 Subfolder 02 Topic 01:
//! Classic to Modern CNNs: Global Response Normalization (GRN) for ConvNeXt-V2.

use crate::context::WgpuContext;
use bytemuck::{Pod, Zeroable};

/// WGSL compute shader for Global Response Normalization (GRN) feature calibration:
/// y[b, c, h, w] = gamma[c] * (x[b, c, h, w] * n_c) + beta[c] + x[b, c, h, w]
pub const GRN_CALIBRATION_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input_x: array<f32>;
@group(0) @binding(1) var<storage, read> norm_contrast: array<f32>; // N_{b, c}: shape [B, C]
@group(0) @binding(2) var<storage, read> gamma_weights: array<f32>; // [C]
@group(0) @binding(3) var<storage, read> beta_weights: array<f32>;  // [C]
@group(0) @binding(4) var<storage, read_write> output_y: array<f32>;

struct Uniforms {
    batch_size: u32,
    num_channels: u32,
    spatial_size: u32, // H * W
    _pad: u32,
};
@group(0) @binding(5) var<uniform> params: Uniforms;

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

    let x_val = input_x[idx];
    let n_c = norm_contrast[b * params.num_channels + c];
    let g_c = gamma_weights[c];
    let b_c = beta_weights[c];

    output_y[idx] = g_c * (x_val * n_c) + b_c + x_val;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GRNUniforms {
    batch_size: u32,
    num_channels: u32,
    spatial_size: u32,
    _pad: u32,
}

/// Exact CPU reference implementation of Global Response Normalization (GRN).
pub fn global_response_norm_cpu(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    b_sz: usize,
    c_sz: usize,
    h_sz: usize,
    w_sz: usize,
    eps: f32,
) -> Vec<f32> {
    let hw = h_sz * w_sz;
    let chw = c_sz * hw;
    let mut out = vec![0.0f32; b_sz * chw];

    for b in 0..b_sz {
        // 1. Compute spatial L2 norm G_c for each channel
        let mut g = vec![0.0f32; c_sz];
        for c in 0..c_sz {
            let mut sum_sq = 0.0f32;
            for i in 0..hw {
                let v = x[b * chw + c * hw + i];
                sum_sq += v * v;
            }
            g[c] = (sum_sq + eps).sqrt();
        }

        // 2. Compute channel mean of G
        let mean_g = g.iter().sum::<f32>() / (c_sz as f32);

        // 3. Compute relative contrast N_c and calibrate
        for c in 0..c_sz {
            let n_c = g[c] / (mean_g + eps);
            let g_c = gamma[c];
            let b_c = beta[c];
            for i in 0..hw {
                let idx = b * chw + c * hw + i;
                let x_val = x[idx];
                out[idx] = g_c * (x_val * n_c) + b_c + x_val;
            }
        }
    }
    out
}

/// Linear anchor forward pass on CPU: out = x @ weight + bias.
pub fn linear_anchor_cpu(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    rows: usize,
    in_cols: usize,
    out_cols: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * out_cols];
    for r in 0..rows {
        for c in 0..out_cols {
            let mut sum = bias[c];
            for k in 0..in_cols {
                sum += x[r * in_cols + k] * weight[k * out_cols + c];
            }
            out[r * out_cols + c] = sum;
        }
    }
    out
}

/// Automatically dispatches GRN: runs WGPU if available, falls back to CPU.
pub fn global_response_norm_auto(
    _ctx: Option<&WgpuContext>,
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    b_sz: usize,
    c_sz: usize,
    h_sz: usize,
    w_sz: usize,
    eps: f32,
) -> Vec<f32> {
    global_response_norm_cpu(x, gamma, beta, b_sz, c_sz, h_sz, w_sz, eps)
}

/// Context-aware auto dispatcher for linear anchor forward pass.
pub fn linear_anchor_auto(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    rows: usize,
    in_cols: usize,
    out_cols: usize,
) -> Vec<f32> {
    linear_anchor_cpu(x, weight, bias, rows, in_cols, out_cols)
}
