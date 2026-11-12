//! Topic 10: Numerical Stability, Welford Variance & LogSumExp (Dual Engine: WGSL + CPU fallback).

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

pub fn welford_mean_var_cpu(x: &[f32], n: usize, d: usize) -> (Vec<f32>, Vec<f32>) {
    let mut mean = vec![0.0f32; d];
    let mut m2 = vec![0.0f32; d];

    for i in 0..n {
        let count = (i + 1) as f32;
        for j in 0..d {
            let val = x[i * d + j];
            let delta = val - mean[j];
            mean[j] += delta / count;
            let delta2 = val - mean[j];
            m2[j] += delta * delta2;
        }
    }

    let mut var = vec![0.0f32; d];
    for j in 0..d {
        var[j] = m2[j] / (n as f32);
    }

    (mean, var)
}

pub fn log_sum_exp_stable_cpu(x: &[f32], n: usize, d: usize) -> Vec<f32> {
    let mut lse_vec = vec![0.0f32; n];
    for i in 0..n {
        let mut max_val = f32::NEG_INFINITY;
        for j in 0..d {
            let val = x[i * d + j];
            if val > max_val {
                max_val = val;
            }
        }
        let mut sum_exp = 0.0f32;
        for j in 0..d {
            sum_exp += (x[i * d + j] - max_val).exp();
        }
        lse_vec[i] = max_val + sum_exp.ln();
    }
    lse_vec
}

pub fn neumaier_sum_2d_cpu(x: &[f32], n: usize, d: usize) -> Vec<f32> {
    let mut sum = vec![0.0f32; d];
    let mut comp = vec![0.0f32; d];

    for i in 0..n {
        for j in 0..d {
            let val = x[i * d + j];
            let t = sum[j] + val;
            let c = if sum[j].abs() >= val.abs() {
                (sum[j] - t) + val
            } else {
                (val - t) + sum[j]
            };
            comp[j] += c;
            sum[j] = t;
        }
    }

    let mut result = vec![0.0f32; d];
    for j in 0..d {
        result[j] = sum[j] + comp[j];
    }
    result
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
