//! Topic 07: Feature Engineering, Tabular Scalers & Weight Folding (Dual Engine: WGSL + CPU fallback).

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

pub fn yeo_johnson_cpu(x: &[f32], lmbda: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len());
    for &val in x {
        if val >= 0.0 {
            if lmbda.abs() > 1e-6 {
                out.push(((val + 1.0).powf(lmbda) - 1.0) / lmbda);
            } else {
                out.push((val + 1.0).ln());
            }
        } else if (lmbda - 2.0).abs() > 1e-6 {
            out.push(-((-val + 1.0).powf(2.0 - lmbda) - 1.0) / (2.0 - lmbda));
        } else {
            out.push(-(-val + 1.0).ln());
        }
    }
    out
}

pub fn fold_weights(
    mean: &[f32],
    std: &[f32],
    w_head: &[f32],
    b_head: &[f32],
    in_dim: usize,
    out_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    // W_fused[i, j] = W_head[i, j] / std[i]
    let mut w_fused = vec![0.0f32; in_dim * out_dim];
    for i in 0..in_dim {
        for j in 0..out_dim {
            w_fused[i * out_dim + j] = w_head[i * out_dim + j] / std[i];
        }
    }

    // b_fused[j] = b_head[j] - sum_i((mean[i] / std[i]) * w_head[i, j])
    let mut b_fused = vec![0.0f32; out_dim];
    for j in 0..out_dim {
        let mut proj = 0.0f32;
        for i in 0..in_dim {
            proj += (mean[i] / std[i]) * w_head[i * out_dim + j];
        }
        b_fused[j] = b_head[j] - proj;
    }

    (w_fused, b_fused)
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
