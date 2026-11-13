//! Topic 09: Non-Euclidean Geometries: Hyperbolic Poincaré, Spherical, and SPD Manifolds (Dual Engine: WGSL + CPU fallback).

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

/// Exponential map from origin on CPU: $\exp_0^c(v) = \tanh(\sqrt{c}\|v\|/2) \frac{v}{\sqrt{c}\|v\|}$.
pub fn poincare_exp_zero_cpu(v: &[f32], batch: usize, dim: usize, c: f32) -> Vec<f32> {
    let sqrt_c = c.sqrt();
    let mut out = vec![0.0f32; batch * dim];

    for i in 0..batch {
        let mut sum_sq = 0.0f32;
        for d in 0..dim {
            let val = v[i * dim + d];
            sum_sq += val * val;
        }
        let v_norm = (sum_sq.sqrt()).max(1e-8);
        let tanh_term = (sqrt_c * v_norm * 0.5).tanh();
        let scale = tanh_term / (sqrt_c * v_norm);

        for d in 0..dim {
            out[i * dim + d] = v[i * dim + d] * scale;
        }
    }
    out
}

/// Logarithmic map to origin on CPU: $\log_0^c(x) = 2 \text{artanh}(\sqrt{c}\|x\|) \frac{x}{\sqrt{c}\|x\|}$.
pub fn poincare_log_zero_cpu(x: &[f32], batch: usize, dim: usize, c: f32) -> Vec<f32> {
    let sqrt_c = c.sqrt();
    let mut out = vec![0.0f32; batch * dim];

    for i in 0..batch {
        let mut sum_sq = 0.0f32;
        for d in 0..dim {
            let val = x[i * dim + d];
            sum_sq += val * val;
        }
        let x_norm = (sum_sq.sqrt()).max(1e-8);
        let clamped_norm = (sqrt_c * x_norm).min(0.9999);
        let atanh_term = 0.5f32 * ((1.0f32 + clamped_norm) / (1.0f32 - clamped_norm)).ln();
        let scale = (2.0f32 * atanh_term) / (sqrt_c * x_norm);

        for d in 0..dim {
            out[i * dim + d] = x[i * dim + d] * scale;
        }
    }
    out
}

/// Möbius addition on CPU: $x \oplus_c y$.
pub fn poincare_mobius_add_cpu(x: &[f32], y: &[f32], batch: usize, dim: usize, c: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * dim];

    for i in 0..batch {
        let mut xy_inner = 0.0f32;
        let mut x_sq = 0.0f32;
        let mut y_sq = 0.0f32;

        for d in 0..dim {
            let x_val = x[i * dim + d];
            let y_val = y[i * dim + d];
            xy_inner += x_val * y_val;
            x_sq += x_val * x_val;
            y_sq += y_val * y_val;
        }

        let two_c_xy = 2.0f32 * c * xy_inner;
        let coef_x = 1.0f32 + two_c_xy + c * y_sq;
        let coef_y = 1.0f32 - c * x_sq;
        let denom = (1.0f32 + two_c_xy + c * c * x_sq * y_sq).max(1e-15);

        for d in 0..dim {
            let num = coef_x * x[i * dim + d] + coef_y * y[i * dim + d];
            out[i * dim + d] = num / denom;
        }
    }
    out
}

/// Geodesic distance in Poincaré Ball on CPU: $\frac{2}{\sqrt{c}}\text{artanh}(\sqrt{c}\|(-x)\oplus_c y\|)$.
pub fn poincare_distance_cpu(x: &[f32], y: &[f32], batch: usize, dim: usize, c: f32) -> Vec<f32> {
    let sqrt_c = c.sqrt();
    let mut neg_x = vec![0.0f32; batch * dim];
    for idx in 0..(batch * dim) {
        neg_x[idx] = -x[idx];
    }

    let diff = poincare_mobius_add_cpu(&neg_x, y, batch, dim, c);
    let mut dist = vec![0.0f32; batch];

    for i in 0..batch {
        let mut sum_sq = 0.0f32;
        for d in 0..dim {
            let val = diff[i * dim + d];
            sum_sq += val * val;
        }
        let diff_norm = (sum_sq.sqrt() * sqrt_c).min(0.9999);
        let atanh_term = 0.5f32 * ((1.0f32 + diff_norm) / (1.0f32 - diff_norm)).ln();
        dist[i] = (2.0f32 / sqrt_c) * atanh_term;
    }
    dist
}

/// Geodesic angular distance on unit sphere on CPU: $\arccos(\langle p, q \rangle)$.
pub fn spherical_distance_cpu(p: &[f32], q: &[f32], batch: usize, dim: usize) -> Vec<f32> {
    let mut dist = vec![0.0f32; batch];
    for i in 0..batch {
        let mut dot = 0.0f32;
        for d in 0..dim {
            dot += p[i * dim + d] * q[i * dim + d];
        }
        let clamped = dot.clamp(-1.0f32 + 1e-7, 1.0f32 - 1e-7);
        dist[i] = clamped.acos();
    }
    dist
}

/// Contrastive alignment loss on unit sphere on CPU: $\|p - q\|_2^2$.
pub fn spherical_alignment_cpu(p: &[f32], q: &[f32], batch: usize, dim: usize) -> Vec<f32> {
    let mut loss = vec![0.0f32; batch];
    for i in 0..batch {
        let mut sum_sq = 0.0f32;
        for d in 0..dim {
            let diff = p[i * dim + d] - q[i * dim + d];
            sum_sq += diff * diff;
        }
        loss[i] = sum_sq;
    }
    loss
}

/// Closed-form matrix logarithm for $2 \times 2$ SPD matrices on CPU.
pub fn spd_matrix_log_2x2_cpu(m: &[f32], batch: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * 4];

    for i in 0..batch {
        let base = i * 4;
        let a = m[base + 0];
        let off = m[base + 1];
        let d = m[base + 3];

        let tr = a + d;
        let delta = ((a - d) * (a - d) + 4.0f32 * off * off).sqrt().max(1e-8);
        let l1 = ((tr + delta) * 0.5f32).max(1e-6);
        let l2 = ((tr - delta) * 0.5f32).max(1e-6);

        let v1_x = l1 - d;
        let v1_y = off;
        let norm1 = (v1_x * v1_x + v1_y * v1_y).sqrt().max(1e-8);
        let u1_x = v1_x / norm1;
        let u1_y = v1_y / norm1;

        let u2_x = -u1_y;
        let u2_y = u1_x;

        let log_l1 = l1.ln();
        let log_l2 = l2.ln();

        let m00 = log_l1 * u1_x * u1_x + log_l2 * u2_x * u2_x;
        let m01 = log_l1 * u1_x * u1_y + log_l2 * u2_x * u2_y;
        let m11 = log_l1 * u1_y * u1_y + log_l2 * u2_y * u2_y;

        out[base + 0] = m00;
        out[base + 1] = m01;
        out[base + 2] = m01;
        out[base + 3] = m11;
    }
    out
}

/// Log-Euclidean distance between $2 \times 2$ SPD matrices on CPU: $\|\log P - \log Q\|_F$.
pub fn spd_log_euclidean_distance_2x2_cpu(p: &[f32], q: &[f32], batch: usize) -> Vec<f32> {
    let log_p = spd_matrix_log_2x2_cpu(p, batch);
    let log_q = spd_matrix_log_2x2_cpu(q, batch);
    let mut dist = vec![0.0f32; batch];

    for i in 0..batch {
        let base = i * 4;
        let mut sum_sq = 0.0f32;
        for k in 0..4 {
            let diff = log_p[base + k] - log_q[base + k];
            sum_sq += diff * diff;
        }
        dist[i] = sum_sq.sqrt();
    }
    dist
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

    let wg_x = (batch as u32).div_ceil(16);
    let wg_y = (out_dim as u32).div_ceil(16);

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
