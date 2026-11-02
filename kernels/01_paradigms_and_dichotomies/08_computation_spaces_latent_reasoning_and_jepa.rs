//! Topic 08: Computation Spaces: Token CoT, Continuous Latent Reasoning, Recurrent Depth, and JEPA (Dual Engine: WGSL + CPU fallback).

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

fn erf_f32(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x_abs = x.abs();
    let p = 0.3275911f32;
    let a1 = 0.254829592f32;
    let a2 = -0.284496736f32;
    let a3 = 1.421413741f32;
    let a4 = -1.453152027f32;
    let a5 = 1.061405429f32;
    let t = 1.0f32 / (1.0f32 + p * x_abs);
    let poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t;
    let result = 1.0f32 - poly * (-x_abs * x_abs).exp();
    sign * result
}

fn gelu_scalar(v: f32) -> f32 {
    const INV_SQRT_2: f32 = 0.7071067811865475;
    0.5f32 * v * (1.0f32 + erf_f32(v * INV_SQRT_2))
}

pub fn rms_norm_cpu(z: &mut [f32], batch: usize, dim: usize, eps: f32) {
    for i in 0..batch {
        let mut sum_sq = 0.0f32;
        for d in 0..dim {
            let val = z[i * dim + d];
            sum_sq += val * val;
        }
        let inv_rms = 1.0f32 / (sum_sq / (dim as f32) + eps).sqrt();
        for d in 0..dim {
            z[i * dim + d] *= inv_rms;
        }
    }
}

/// Evaluates K-step input-anchored recurrent latent deliberation on CPU.
pub fn continuous_latent_reasoning_cpu(
    x: &[f32],
    w_proj: &[f32],
    b_proj: &[f32],
    w_z: &[f32],
    w_anc: &[f32],
    b_z: &[f32],
    batch: usize,
    in_dim: usize,
    dim: usize,
    steps: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut e_x = linear_forward_cpu(x, w_proj, b_proj, batch, in_dim, dim);
    for v in &mut e_x {
        *v = gelu_scalar(*v);
    }

    let mut z = e_x.clone();
    let mut step_history = Vec::with_capacity(steps + 1);
    step_history.push(z.clone());

    for _ in 0..steps {
        let mut next_z = vec![0.0f32; batch * dim];
        for i in 0..batch {
            for d in 0..dim {
                let mut acc = b_z[d];
                for k in 0..dim {
                    acc +=
                        z[i * dim + k] * w_z[k * dim + d] + e_x[i * dim + k] * w_anc[k * dim + d];
                }
                next_z[i * dim + d] = z[i * dim + d] + gelu_scalar(acc);
            }
        }
        rms_norm_cpu(&mut next_z, batch, dim, 1e-5);
        z = next_z;
        step_history.push(z.clone());
    }

    let mut trajectories = vec![0.0f32; batch * (steps + 1) * dim];
    for i in 0..batch {
        for s in 0..=steps {
            for d in 0..dim {
                trajectories[i * (steps + 1) * dim + s * dim + d] = step_history[s][i * dim + d];
            }
        }
    }

    (trajectories, z)
}

/// Evaluates adaptive computation halting probabilities on CPU.
pub fn ponder_halting_cpu(
    trajectories: &[f32],
    w_h: &[f32],
    b_h: &[f32],
    batch: usize,
    steps: usize,
    dim: usize,
) -> Vec<f32> {
    let mut halting_probs = vec![0.0f32; batch * steps];

    for i in 0..batch {
        let mut unhalted = 1.0f32;
        for k in 0..steps {
            let offset = i * (steps + 1) * dim + (k + 1) * dim;
            let mut acc = b_h[0];
            for d in 0..dim {
                acc += trajectories[offset + d] * w_h[d];
            }
            let lam = 1.0f32 / (1.0f32 + (-acc).exp());
            let p_k = lam * unhalted;
            halting_probs[i * steps + k] = p_k;
            unhalted *= 1.0f32 - lam;
        }
    }

    halting_probs
}

/// Evaluates JEPA latent-space prediction on CPU.
pub fn jepa_predict_cpu(
    s_x: &[f32],
    w_p1: &[f32],
    b_p1: &[f32],
    w_p2: &[f32],
    b_p2: &[f32],
    batch: usize,
    dim: usize,
) -> Vec<f32> {
    let mut h = linear_forward_cpu(s_x, w_p1, b_p1, batch, dim, dim);
    for v in &mut h {
        *v = gelu_scalar(*v);
    }
    linear_forward_cpu(&h, w_p2, b_p2, batch, dim, dim)
}

/// Evaluates squared L2 representation prediction loss.
pub fn jepa_loss_cpu(pred: &[f32], target: &[f32], batch: usize, dim: usize) -> Vec<f32> {
    let mut loss = vec![0.0f32; batch];
    for i in 0..batch {
        let mut sum_sq = 0.0f32;
        for d in 0..dim {
            let diff = pred[i * dim + d] - target[i * dim + d];
            sum_sq += diff * diff;
        }
        loss[i] = sum_sq;
    }
    loss
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
