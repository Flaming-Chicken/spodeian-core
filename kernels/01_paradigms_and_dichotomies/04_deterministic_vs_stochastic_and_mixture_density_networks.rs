//! Topic 04: Deterministic vs Stochastic & Mixture Density Networks (Dual Engine: WGSL + CPU fallback).

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

/// Evaluates deterministic 2-layer regression forward pass on CPU.
pub fn deterministic_forward_cpu(
    x: &[f32],
    w_h: &[f32],
    b_h: &[f32],
    w_det: &[f32],
    b_det: &[f32],
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
            let mut acc = b_h[hd];
            for k in 0..in_dim {
                acc += x_i[k] * w_h[k * hidden_dim + hd];
            }
            h[hd] = acc.max(0.0f32);
        }
        for j in 0..out_dim {
            let mut acc = b_det[j];
            for hd in 0..hidden_dim {
                acc += h[hd] * w_det[hd * out_dim + j];
            }
            out[i * out_dim + j] = acc;
        }
    }
    out
}

/// Evaluates Mixture Density Network parameter extraction on CPU: (pi, mu, sigma).
pub fn mdn_forward_cpu(
    x: &[f32],
    w_h: &[f32],
    b_h: &[f32],
    w_pi: &[f32],
    b_pi: &[f32],
    w_mu: &[f32],
    b_mu: &[f32],
    w_sigma: &[f32],
    b_sigma: &[f32],
    batch: usize,
    in_dim: usize,
    hidden_dim: usize,
    num_components: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut pi = vec![0.0f32; batch * num_components];
    let mut mu = vec![0.0f32; batch * num_components];
    let mut sigma = vec![0.0f32; batch * num_components];

    for i in 0..batch {
        let x_i = &x[i * in_dim..(i + 1) * in_dim];
        let mut h = vec![0.0f32; hidden_dim];
        for hd in 0..hidden_dim {
            let mut acc = b_h[hd];
            for k in 0..in_dim {
                acc += x_i[k] * w_h[k * hidden_dim + hd];
            }
            h[hd] = acc.max(0.0f32);
        }

        let mut pi_logits = vec![0.0f32; num_components];
        let mut max_logit = f32::NEG_INFINITY;
        for c in 0..num_components {
            let mut acc = b_pi[c];
            for hd in 0..hidden_dim {
                acc += h[hd] * w_pi[hd * num_components + c];
            }
            pi_logits[c] = acc;
            if acc > max_logit {
                max_logit = acc;
            }
        }
        let mut sum_exp = 0.0f32;
        for c in 0..num_components {
            let exp_val = (pi_logits[c] - max_logit).exp();
            pi[i * num_components + c] = exp_val;
            sum_exp += exp_val;
        }
        let inv_sum = 1.0f32 / sum_exp.max(1e-12);
        for c in 0..num_components {
            pi[i * num_components + c] *= inv_sum;
        }

        for c in 0..num_components {
            let mut acc_mu = b_mu[c];
            let mut acc_sig = b_sigma[c];
            for hd in 0..hidden_dim {
                acc_mu += h[hd] * w_mu[hd * num_components + c];
                acc_sig += h[hd] * w_sigma[hd * num_components + c];
            }
            mu[i * num_components + c] = acc_mu;
            sigma[i * num_components + c] = acc_sig.exp();
        }
    }

    (pi, mu, sigma)
}

/// Evaluates MDN log-likelihood log p(y|x) on CPU.
pub fn mdn_log_prob_cpu(
    pi: &[f32],
    mu: &[f32],
    sigma: &[f32],
    y: &[f32],
    batch: usize,
    num_components: usize,
) -> Vec<f32> {
    const SQRT_2PI: f32 = 2.5066282746310002;
    let mut log_probs = vec![0.0f32; batch];

    for i in 0..batch {
        let y_val = y[i];
        let mut sum_prob = 0.0f32;
        for c in 0..num_components {
            let idx = i * num_components + c;
            let p_c = pi[idx];
            let m_c = mu[idx];
            let s_c = sigma[idx].max(1e-6);

            let diff = y_val - m_c;
            let normal_pdf = (-0.5f32 * (diff / s_c).powi(2)).exp() / (s_c * SQRT_2PI);
            sum_prob += p_c * normal_pdf;
        }
        log_probs[i] = sum_prob.max(1e-12).ln();
    }

    log_probs
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
