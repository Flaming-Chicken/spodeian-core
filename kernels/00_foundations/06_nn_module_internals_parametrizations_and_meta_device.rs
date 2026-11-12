//! Topic 06: Module Internals & Cayley Orthogonal Parametrizations (Dual Engine: WGSL + CPU fallback).

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

pub fn invert_matrix(p: &[f32], n: usize) -> Vec<f32> {
    let mut aug = vec![0.0f32; n * (2 * n)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (2 * n) + j] = p[i * n + j];
        }
        aug[i * (2 * n) + n + i] = 1.0;
    }

    for i in 0..n {
        let mut pivot_row = i;
        for k in (i + 1)..n {
            if aug[k * (2 * n) + i].abs() > aug[pivot_row * (2 * n) + i].abs() {
                pivot_row = k;
            }
        }
        if pivot_row != i {
            for c in 0..(2 * n) {
                let tmp = aug[i * (2 * n) + c];
                aug[i * (2 * n) + c] = aug[pivot_row * (2 * n) + c];
                aug[pivot_row * (2 * n) + c] = tmp;
            }
        }
        let pivot = aug[i * (2 * n) + i];
        assert!(pivot.abs() > 1e-12, "Singular matrix in inversion");
        for j in 0..(2 * n) {
            aug[i * (2 * n) + j] /= pivot;
        }
        for r in 0..n {
            if r != i {
                let factor = aug[r * (2 * n) + i];
                for c in 0..(2 * n) {
                    aug[r * (2 * n) + c] -= factor * aug[i * (2 * n) + c];
                }
            }
        }
    }

    let mut inv = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = aug[i * (2 * n) + n + j];
        }
    }
    inv
}

pub fn cayley_transform(m: &[f32], n: usize) -> Vec<f32> {
    // p = I + A, q = I - A where A is skew-symmetric: A_ij = m_ij, A_ji = -m_ij for j > i
    let mut p = vec![0.0f32; n * n];
    let mut q = vec![0.0f32; n * n];
    for i in 0..n {
        p[i * n + i] = 1.0;
        q[i * n + i] = 1.0;
        for j in (i + 1)..n {
            let val = m[i * n + j];
            p[i * n + j] = val;
            p[j * n + i] = -val;
            q[i * n + j] = -val;
            q[j * n + i] = val;
        }
    }

    let p_inv = invert_matrix(&p, n);
    // W = q @ p_inv
    let mut w = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0f32;
            for k in 0..n {
                acc += q[i * n + k] * p_inv[k * n + j];
            }
            w[i * n + j] = acc;
        }
    }
    w
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
