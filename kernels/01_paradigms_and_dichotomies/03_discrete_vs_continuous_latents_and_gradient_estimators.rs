//! Topic 03: Discrete vs Continuous Latents & Gradient Estimators (Dual Engine: WGSL + CPU fallback).

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

/// Gaussian encoder forward pass on CPU.
pub fn gaussian_encode_cpu(
    x: &[f32],
    w_mu: &[f32],
    b_mu: &[f32],
    w_logvar: &[f32],
    b_logvar: &[f32],
    batch: usize,
    in_dim: usize,
    latent_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mu = linear_forward_cpu(x, w_mu, b_mu, batch, in_dim, latent_dim);
    let logvar = linear_forward_cpu(x, w_logvar, b_logvar, batch, in_dim, latent_dim);
    (mu, logvar)
}

/// Reparameterization trick: z = mu + exp(0.5 * logvar) * eps.
pub fn gaussian_reparameterize_cpu(
    mu: &[f32],
    logvar: &[f32],
    eps: &[f32],
    batch: usize,
    latent_dim: usize,
) -> Vec<f32> {
    let mut z = vec![0.0f32; batch * latent_dim];
    for i in 0..batch * latent_dim {
        let std = (0.5f32 * logvar[i]).exp();
        z[i] = mu[i] + std * eps[i];
    }
    z
}

/// Analytical KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar)).
pub fn gaussian_kl_divergence_cpu(
    mu: &[f32],
    logvar: &[f32],
    batch: usize,
    latent_dim: usize,
) -> Vec<f32> {
    let mut kl = vec![0.0f32; batch];
    for i in 0..batch {
        let mut sum_kl = 0.0f32;
        for d in 0..latent_dim {
            let idx = i * latent_dim + d;
            let m = mu[idx];
            let lv = logvar[idx];
            sum_kl += 1.0f32 + lv - m * m - lv.exp();
        }
        kl[i] = -0.5f32 * sum_kl;
    }
    kl
}

/// Linear decoder forward pass on CPU.
pub fn linear_decode_cpu(
    z: &[f32],
    w_dec: &[f32],
    b_dec: &[f32],
    batch: usize,
    latent_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    linear_forward_cpu(z, w_dec, b_dec, batch, latent_dim, out_dim)
}

/// Gumbel-Softmax sample: softmax((logits + g) / temp).
pub fn gumbel_softmax_sample_cpu(
    x: &[f32],
    w_logits: &[f32],
    b_logits: &[f32],
    gumbel_noise: &[f32],
    temperature: f32,
    batch: usize,
    in_dim: usize,
    num_classes: usize,
) -> Vec<f32> {
    let logits = linear_forward_cpu(x, w_logits, b_logits, batch, in_dim, num_classes);
    let mut relaxed = vec![0.0f32; batch * num_classes];
    for i in 0..batch {
        let row_start = i * num_classes;
        let mut max_val = f32::NEG_INFINITY;
        for c in 0..num_classes {
            let val = (logits[row_start + c] + gumbel_noise[row_start + c]) / temperature;
            relaxed[row_start + c] = val;
            if val > max_val {
                max_val = val;
            }
        }
        let mut sum_exp = 0.0f32;
        for c in 0..num_classes {
            let exp_val = (relaxed[row_start + c] - max_val).exp();
            relaxed[row_start + c] = exp_val;
            sum_exp += exp_val;
        }
        let inv_sum = 1.0f32 / sum_exp.max(1e-12);
        for c in 0..num_classes {
            relaxed[row_start + c] *= inv_sum;
        }
    }
    relaxed
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
