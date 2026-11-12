//! Topic 02: Tree vs Kernel vs Neural Inductive Biases (Dual Engine: WGSL + CPU fallback).

use crate::context::{KernelDevice, WgpuContext};

pub const WGSL_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

struct Dims {
    batch: u32,
    in_dim: u32,
    out_dim: u32,
};
@group(0) @binding(4) var<uniform> dims: Dims;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    let col = id.y;
    if (row >= dims.batch || col >= dims.out_dim) {
        return;
    }
    var acc: f32 = b[col];
    for (var i: u32 = 0u; i < dims.in_dim; i = i + 1u) {
        acc = acc + x[row * dims.in_dim + i] * w[i * dims.out_dim + col];
    }
    y[row * dims.out_dim + col] = acc;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Dims {
    batch: u32,
    in_dim: u32,
    out_dim: u32,
}

pub fn linear_forward_cpu(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    batch: usize,
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    let mut y = vec![0.0f32; batch * out_dim];
    for i in 0..batch {
        for j in 0..out_dim {
            let mut acc = b[j];
            for k in 0..in_dim {
                acc += x[i * in_dim + k] * w[k * out_dim + j];
            }
            y[i * out_dim + j] = acc;
        }
    }
    y
}

/// Evaluates a depth-2 soft decision tree on CPU.
pub fn soft_decision_tree_cpu(
    x: &[f32],
    routing_weights: &[f32],
    thresholds: &[f32],
    temperature: f32,
    leaf_weights: &[f32],
    batch: usize,
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * out_dim];
    for i in 0..batch {
        let x_i = &x[i * in_dim..(i + 1) * in_dim];
        let mut p_right = [0.0f32; 3];
        for node in 0..3 {
            let mut logit = 0.0f32;
            for k in 0..in_dim {
                logit += x_i[k] * routing_weights[node * in_dim + k];
            }
            logit = temperature * (logit - thresholds[node]);
            p_right[node] = 1.0f32 / (1.0f32 + (-logit).exp());
        }
        let p_left = [1.0 - p_right[0], 1.0 - p_right[1], 1.0 - p_right[2]];

        let mu = [
            p_left[0] * p_left[1],
            p_left[0] * p_right[1],
            p_right[0] * p_left[2],
            p_right[0] * p_right[2],
        ];

        for j in 0..out_dim {
            let mut acc = 0.0f32;
            for leaf in 0..4 {
                acc += mu[leaf] * leaf_weights[leaf * out_dim + j];
            }
            out[i * out_dim + j] = acc;
        }
    }
    out
}

/// Evaluates RBF kernel ridge regression on CPU.
pub fn rbf_kernel_ridge_cpu(
    x: &[f32],
    support: &[f32],
    dual_coeffs: &[f32],
    gamma: f32,
    bias: &[f32],
    batch: usize,
    supp_len: usize,
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * out_dim];
    for i in 0..batch {
        let x_i = &x[i * in_dim..(i + 1) * in_dim];
        for j in 0..out_dim {
            let mut acc = bias[j];
            for s in 0..supp_len {
                let supp_s = &support[s * in_dim..(s + 1) * in_dim];
                let mut dist_sq = 0.0f32;
                for k in 0..in_dim {
                    let d = x_i[k] - supp_s[k];
                    dist_sq += d * d;
                }
                let k_val = (-gamma * dist_sq).exp();
                acc += k_val * dual_coeffs[s * out_dim + j];
            }
            out[i * out_dim + j] = acc;
        }
    }
    out
}

/// Evaluates 2-layer ReLU MLP on CPU.
pub fn mlp_classifier_cpu(
    x: &[f32],
    w1: &[f32],
    b1: &[f32],
    w2: &[f32],
    b2: &[f32],
    batch: usize,
    in_dim: usize,
    hidden_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * out_dim];
    for i in 0..batch {
        let x_i = &x[i * in_dim..(i + 1) * in_dim];
        let mut h = vec![0.0f32; hidden_dim];
        for hd in 0..hidden_dim {
            let mut acc = b1[hd];
            for k in 0..in_dim {
                acc += x_i[k] * w1[k * hidden_dim + hd];
            }
            h[hd] = acc.max(0.0f32);
        }
        for j in 0..out_dim {
            let mut acc = b2[j];
            for hd in 0..hidden_dim {
                acc += h[hd] * w2[hd * out_dim + j];
            }
            out[i * out_dim + j] = acc;
        }
    }
    out
}

#[cfg(feature = "wgpu")]
pub fn linear_forward_wgpu(
    ctx: &WgpuContext,
    x: &[f32],
    w: &[f32],
    b: &[f32],
    batch: usize,
    in_dim: usize,
    out_dim: usize,
) -> Option<Vec<f32>> {
    let dims = Dims {
        batch: batch as u32,
        in_dim: in_dim as u32,
        out_dim: out_dim as u32,
    };

    let wg_x = ((batch as u32) + 15) / 16;
    let wg_y = ((out_dim as u32) + 15) / 16;

    let bytes = crate::wgpu_util::run_compute_shader(
        ctx,
        WGSL_SHADER,
        "main",
        &[
            bytemuck::cast_slice(x),
            bytemuck::cast_slice(w),
            bytemuck::cast_slice(b),
        ],
        batch * out_dim * std::mem::size_of::<f32>(),
        Some(bytemuck::bytes_of(&dims)),
        (wg_x, wg_y, 1),
    )?;

    Some(bytemuck::cast_slice::<u8, f32>(&bytes).to_vec())
}

#[cfg(not(feature = "wgpu"))]
pub fn linear_forward_wgpu(
    _ctx: &WgpuContext,
    _x: &[f32],
    _w: &[f32],
    _b: &[f32],
    _batch: usize,
    _in_dim: usize,
    _out_dim: usize,
) -> Option<Vec<f32>> {
    None
}

pub fn linear_forward(
    device: &KernelDevice,
    x: &[f32],
    w: &[f32],
    b: &[f32],
    batch: usize,
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    match device {
        KernelDevice::Wgpu(ctx) => {
            if let Some(res) = linear_forward_wgpu(ctx, x, w, b, batch, in_dim, out_dim) {
                return res;
            }
            linear_forward_cpu(x, w, b, batch, in_dim, out_dim)
        }
        KernelDevice::Cpu => linear_forward_cpu(x, w, b, batch, in_dim, out_dim),
    }
}
