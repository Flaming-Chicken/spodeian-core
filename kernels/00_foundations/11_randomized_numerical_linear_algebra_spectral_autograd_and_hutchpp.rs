//! Topic 11: Randomized Linear Algebra, Spectral Autograd & Hutch++ (Dual Engine: WGSL + CPU fallback).

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

pub fn modified_gram_schmidt_qr_cpu(y: &[f32], m: usize, l: usize) -> (Vec<f32>, Vec<f32>) {
    let mut q_cols: Vec<Vec<f32>> = (0..l)
        .map(|j| (0..m).map(|i| y[i * l + j]).collect())
        .collect();
    let mut r_mat = vec![vec![0.0f32; l]; l];

    for j in 0..l {
        for i in 0..j {
            let mut dot = 0.0f32;
            for k in 0..m {
                dot += q_cols[i][k] * q_cols[j][k];
            }
            r_mat[i][j] = dot;
            for k in 0..m {
                q_cols[j][k] -= dot * q_cols[i][k];
            }
        }
        let mut norm_sq = 0.0f32;
        for k in 0..m {
            norm_sq += q_cols[j][k] * q_cols[j][k];
        }
        let norm = norm_sq.sqrt();
        r_mat[j][j] = norm;
        let scale = 1.0f32 / (norm + 1e-12);
        for k in 0..m {
            q_cols[j][k] *= scale;
        }
    }

    let mut q_mat = vec![0.0f32; m * l];
    for i in 0..m {
        for j in 0..l {
            q_mat[i * l + j] = q_cols[j][i];
        }
    }

    let mut r_flat = vec![0.0f32; l * l];
    for i in 0..l {
        for j in 0..l {
            r_flat[i * l + j] = r_mat[i][j];
        }
    }

    (q_mat, r_flat)
}

pub fn power_iteration_spectral_norm_cpu(
    w: &[f32],
    v_init: &[f32],
    m: usize,
    n: usize,
    num_iters: usize,
) -> (f32, Vec<f32>, Vec<f32>) {
    let mut v = v_init.to_vec();
    let mut norm_sq = 0.0f32;
    for &val in &v {
        norm_sq += val * val;
    }
    let norm_v = norm_sq.sqrt();
    for val in &mut v {
        *val /= norm_v + 1e-12;
    }

    let mut u = vec![0.0f32; m];

    for _ in 0..num_iters {
        // u = W @ v  [m]
        let mut u_unnorm = vec![0.0f32; m];
        for i in 0..m {
            let mut acc = 0.0f32;
            for j in 0..n {
                acc += w[i * n + j] * v[j];
            }
            u_unnorm[i] = acc;
        }
        let mut norm_u_sq = 0.0f32;
        for &val in &u_unnorm {
            norm_u_sq += val * val;
        }
        let norm_u = norm_u_sq.sqrt();
        for i in 0..m {
            u[i] = u_unnorm[i] / (norm_u + 1e-12);
        }

        // v = W^T @ u [n]
        let mut v_unnorm = vec![0.0f32; n];
        for j in 0..n {
            let mut acc = 0.0f32;
            for i in 0..m {
                acc += w[i * n + j] * u[i];
            }
            v_unnorm[j] = acc;
        }
        let mut norm_v_sq = 0.0f32;
        for &val in &v_unnorm {
            norm_v_sq += val * val;
        }
        let norm_v_next = norm_v_sq.sqrt();
        for j in 0..n {
            v[j] = v_unnorm[j] / (norm_v_next + 1e-12);
        }
    }

    // sigma = u^T @ W @ v
    let mut w_v = vec![0.0f32; m];
    for i in 0..m {
        let mut acc = 0.0f32;
        for j in 0..n {
            acc += w[i * n + j] * v[j];
        }
        w_v[i] = acc;
    }
    let mut sigma = 0.0f32;
    for i in 0..m {
        sigma += u[i] * w_v[i];
    }

    (sigma, u, v)
}

pub fn lorentzian_broadened_f_matrix_cpu(s: &[f32], epsilon: f32) -> Vec<f32> {
    let n = s.len();
    let eps_sq = epsilon * epsilon;
    let mut f_data = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let delta = s[j] - s[i];
                f_data[i * n + j] = delta / (delta * delta + eps_sq);
            }
        }
    }
    f_data
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
