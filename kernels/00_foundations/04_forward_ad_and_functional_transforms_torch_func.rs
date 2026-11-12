//! Topic 04: Forward AD & Functional Transforms (Dual Engine: WGSL + CPU fallback).

use crate::context::{KernelDevice, WgpuContext};

pub const WGSL_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> vx: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(5) var<storage, read_write> vy: array<f32>;

struct Dims {
    batch: u32,
    in_dim: u32,
    out_dim: u32,
};
@group(0) @binding(6) var<uniform> dims: Dims;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    let col = id.y;
    if (row >= dims.batch || col >= dims.out_dim) {
        return;
    }
    var acc_y: f32 = b[col];
    var acc_vy: f32 = 0.0;
    for (var i: u32 = 0u; i < dims.in_dim; i = i + 1u) {
        let weight = w[i * dims.out_dim + col];
        acc_y = acc_y + x[row * dims.in_dim + i] * weight;
        acc_vy = acc_vy + vx[row * dims.in_dim + i] * weight;
    }
    y[row * dims.out_dim + col] = acc_y;
    vy[row * dims.out_dim + col] = acc_vy;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Dims {
    batch: u32,
    in_dim: u32,
    out_dim: u32,
}

pub fn linear_jvp_cpu(
    x: &[f32],
    vx: &[f32],
    w: &[f32],
    b: &[f32],
    batch: usize,
    in_dim: usize,
    out_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut y = vec![0.0f32; batch * out_dim];
    let mut vy = vec![0.0f32; batch * out_dim];
    for i in 0..batch {
        for j in 0..out_dim {
            let mut acc_y = b[j];
            let mut acc_vy = 0.0f32;
            for k in 0..in_dim {
                let weight = w[k * out_dim + j];
                acc_y += x[i * in_dim + k] * weight;
                acc_vy += vx[i * in_dim + k] * weight;
            }
            y[i * out_dim + j] = acc_y;
            vy[i * out_dim + j] = acc_vy;
        }
    }
    (y, vy)
}

#[cfg(feature = "wgpu")]
pub fn linear_jvp_wgpu(
    ctx: &WgpuContext,
    x: &[f32],
    vx: &[f32],
    w: &[f32],
    b: &[f32],
    batch: usize,
    in_dim: usize,
    out_dim: usize,
) -> Option<(Vec<f32>, Vec<f32>)> {
    let dims = Dims {
        batch: batch as u32,
        in_dim: in_dim as u32,
        out_dim: out_dim as u32,
    };
    let wg_x = ((batch as u32) + 15) / 16;
    let wg_y = ((out_dim as u32) + 15) / 16;

    // Use single output buffer packing both y and vy to reuse run_compute_shader
    // WGSL compute pipeline layout:
    let packed_shader = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> vx: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> out_packed: array<f32>;

struct Dims {
    batch: u32,
    in_dim: u32,
    out_dim: u32,
};
@group(0) @binding(5) var<uniform> dims: Dims;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    let col = id.y;
    if (row >= dims.batch || col >= dims.out_dim) {
        return;
    }
    var acc_y: f32 = b[col];
    var acc_vy: f32 = 0.0;
    for (var i: u32 = 0u; i < dims.in_dim; i = i + 1u) {
        let weight = w[i * dims.out_dim + col];
        acc_y = acc_y + x[row * dims.in_dim + i] * weight;
        acc_vy = acc_vy + vx[row * dims.in_dim + i] * weight;
    }
    let total = dims.batch * dims.out_dim;
    let idx = row * dims.out_dim + col;
    out_packed[idx] = acc_y;
    out_packed[total + idx] = acc_vy;
}
"#;

    let output_elems = 2 * batch * out_dim;
    let bytes = crate::wgpu_util::run_compute_shader(
        ctx,
        packed_shader,
        "main",
        &[
            bytemuck::cast_slice(x),
            bytemuck::cast_slice(vx),
            bytemuck::cast_slice(w),
            bytemuck::cast_slice(b),
        ],
        output_elems * std::mem::size_of::<f32>(),
        Some(bytemuck::bytes_of(&dims)),
        (wg_x, wg_y, 1),
    )?;

    let floats: &[f32] = bytemuck::cast_slice(&bytes);
    let n = batch * out_dim;
    let y = floats[..n].to_vec();
    let vy = floats[n..2 * n].to_vec();
    Some((y, vy))
}

#[cfg(not(feature = "wgpu"))]
pub fn linear_jvp_wgpu(
    _ctx: &WgpuContext,
    _x: &[f32],
    _vx: &[f32],
    _w: &[f32],
    _b: &[f32],
    _batch: usize,
    _in_dim: usize,
    _out_dim: usize,
) -> Option<(Vec<f32>, Vec<f32>)> {
    None
}

pub fn linear_jvp(
    device: &KernelDevice,
    x: &[f32],
    vx: &[f32],
    w: &[f32],
    b: &[f32],
    batch: usize,
    in_dim: usize,
    out_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    match device {
        KernelDevice::Wgpu(ctx) => {
            if let Some(res) = linear_jvp_wgpu(ctx, x, vx, w, b, batch, in_dim, out_dim) {
                return res;
            }
            linear_jvp_cpu(x, vx, w, b, batch, in_dim, out_dim)
        }
        KernelDevice::Cpu => linear_jvp_cpu(x, vx, w, b, batch, in_dim, out_dim),
    }
}
