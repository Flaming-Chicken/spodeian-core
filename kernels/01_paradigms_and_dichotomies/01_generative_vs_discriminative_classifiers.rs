//! Topic 01: Generative vs Discriminative Classifiers (Dual Engine: WGSL + CPU fallback).

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

/// Evaluates affine discriminant logits on CPU: x @ w + b.
pub fn predict_logits_gda_cpu(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    batch: usize,
    in_dim: usize,
    num_classes: usize,
) -> Vec<f32> {
    let mut logits = vec![0.0f32; batch * num_classes];
    for i in 0..batch {
        for c in 0..num_classes {
            let mut acc = b[c];
            for k in 0..in_dim {
                acc += x[i * in_dim + k] * w[k * num_classes + c];
            }
            logits[i * num_classes + c] = acc;
        }
    }
    logits
}

/// Evaluates row-wise numerically stable softmax on CPU.
pub fn stable_softmax_cpu(logits: &[f32], batch: usize, num_classes: usize) -> Vec<f32> {
    let mut probs = vec![0.0f32; batch * num_classes];
    for i in 0..batch {
        let row_start = i * num_classes;
        let mut max_val = f32::NEG_INFINITY;
        for c in 0..num_classes {
            let val = logits[row_start + c];
            if val > max_val {
                max_val = val;
            }
        }
        let mut sum_exp = 0.0f32;
        for c in 0..num_classes {
            let exp_val = (logits[row_start + c] - max_val).exp();
            probs[row_start + c] = exp_val;
            sum_exp += exp_val;
        }
        let inv_sum = 1.0f32 / sum_exp.max(1e-12);
        for c in 0..num_classes {
            probs[row_start + c] *= inv_sum;
        }
    }
    probs
}

/// Evaluates discriminative linear forward pass on CPU.
pub fn discriminative_forward_cpu(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    batch: usize,
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    predict_logits_gda_cpu(x, w, b, batch, in_dim, out_dim)
}

/// WebGPU WGSL compute shader execution for affine projection.
#[cfg(feature = "wgpu")]
pub fn affine_forward_wgpu(
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
pub fn affine_forward_wgpu(
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

/// Unified entrypoint for GDA logits evaluation.
pub fn predict_logits_gda(
    device: &KernelDevice,
    x: &[f32],
    w: &[f32],
    b: &[f32],
    batch: usize,
    in_dim: usize,
    num_classes: usize,
) -> Vec<f32> {
    match device {
        KernelDevice::Wgpu(ctx) => {
            if let Some(res) = affine_forward_wgpu(ctx, x, w, b, batch, in_dim, num_classes) {
                return res;
            }
            predict_logits_gda_cpu(x, w, b, batch, in_dim, num_classes)
        }
        KernelDevice::Cpu => predict_logits_gda_cpu(x, w, b, batch, in_dim, num_classes),
    }
}

/// Unified entrypoint for discriminative linear forward.
pub fn discriminative_forward(
    device: &KernelDevice,
    x: &[f32],
    w: &[f32],
    b: &[f32],
    batch: usize,
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    predict_logits_gda(device, x, w, b, batch, in_dim, out_dim)
}
