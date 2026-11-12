//! Topic 05: Parametric vs Nonparametric & Neural Processes (Dual Engine: WGSL + CPU fallback).

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

/// Evaluates parametric 2-layer MLP on CPU: ReLU(x @ w1 + b1) @ w2 + b2.
pub fn parametric_forward_cpu(
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

/// Computes pairwise RBF kernel on CPU: exp(-0.5 * ||x1 - x2||^2 / lengthscale^2).
pub fn gp_rbf_kernel_cpu(
    x1: &[f32],
    x2: &[f32],
    lengthscale: f32,
    batch: usize,
    n1: usize,
    n2: usize,
    dim: usize,
) -> Vec<f32> {
    let ls_sq = lengthscale * lengthscale;
    let mut k_mat = vec![0.0f32; batch * n1 * n2];

    for b in 0..batch {
        let b_offset1 = b * n1 * dim;
        let b_offset2 = b * n2 * dim;
        let k_offset = b * n1 * n2;

        for i in 0..n1 {
            let p1 = &x1[b_offset1 + i * dim..b_offset1 + (i + 1) * dim];
            for j in 0..n2 {
                let p2 = &x2[b_offset2 + j * dim..b_offset2 + (j + 1) * dim];
                let mut dist_sq = 0.0f32;
                for d in 0..dim {
                    let diff = p1[d] - p2[d];
                    dist_sq += diff * diff;
                }
                k_mat[k_offset + i * n2 + j] = (-0.5f32 * dist_sq / ls_sq).exp();
            }
        }
    }
    k_mat
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
