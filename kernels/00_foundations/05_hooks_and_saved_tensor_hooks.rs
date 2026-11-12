//! Topic 05: Hooks & Saved Tensor Hooks (Dual Engine: WGSL + CPU fallback).

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

pub fn linear_backward_cpu(
    saved_x: &[f32],
    pred: &[f32],
    target: &[f32],
    w: &[f32],
    batch: usize,
    in_dim: usize,
    out_dim: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // d_pred = pred - target
    let mut d_pred = vec![0.0f32; batch * out_dim];
    for i in 0..batch * out_dim {
        d_pred[i] = pred[i] - target[i];
    }

    // grad_x = d_pred @ w^T  [batch, in_dim]
    let mut grad_x = vec![0.0f32; batch * in_dim];
    for i in 0..batch {
        for j in 0..in_dim {
            let mut acc = 0.0f32;
            for k in 0..out_dim {
                acc += d_pred[i * out_dim + k] * w[j * out_dim + k];
            }
            grad_x[i * in_dim + j] = acc;
        }
    }

    // grad_w = saved_x^T @ d_pred [in_dim, out_dim]
    let mut grad_w = vec![0.0f32; in_dim * out_dim];
    for i in 0..in_dim {
        for j in 0..out_dim {
            let mut acc = 0.0f32;
            for k in 0..batch {
                acc += saved_x[k * in_dim + i] * d_pred[k * out_dim + j];
            }
            grad_w[i * out_dim + j] = acc;
        }
    }

    // grad_b = sum_batch(d_pred) [out_dim]
    let mut grad_b = vec![0.0f32; out_dim];
    for j in 0..out_dim {
        let mut acc = 0.0f32;
        for i in 0..batch {
            acc += d_pred[i * out_dim + j];
        }
        grad_b[j] = acc;
    }

    (grad_x, grad_w, grad_b)
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
