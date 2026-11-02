//! Hardware Kernels (WGSL + Native CPU fallback) for Category 04 Topic 08:
//! 01_constrained_proximal_frank_wolfe_admm_and_mirror_descent in 04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation.

use crate::context::KernelDevice;

/// WGSL Compute Shader for Proximal Gradient Soft-Thresholding.
pub const ISTA_SOFT_THRESH_SHADER: &str = r#"
struct IstaUniforms {
    len: u32,
    lr: f32,
    thresh: f32,
    _pad0: f32,
};

@group(0) @binding(0) var<uniform> uniforms: IstaUniforms;
@group(0) @binding(1) var<storage, read> param_in: array<f32>;
@group(0) @binding(2) var<storage, read> grad_in: array<f32>;
@group(0) @binding(3) var<storage, read_write> param_out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.len) {
        return;
    }
    let v = param_in[idx] - uniforms.lr * grad_in[idx];
    let abs_v = abs(v);
    let sign_v = sign(v);
    let shrunk = max(abs_v - uniforms.thresh, 0.0);
    param_out[idx] = sign_v * shrunk;
}
"#;

/// WGSL Compute Shader for PGD Projection Scale.
pub const PGD_SCALE_SHADER: &str = r#"
struct PgdUniforms {
    len: u32,
    scale: f32,
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0) var<uniform> uniforms: PgdUniforms;
@group(0) @binding(1) var<storage, read> v_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> param_out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.len) {
        return;
    }
    param_out[idx] = v_in[idx] * uniforms.scale;
}
"#;

/// CPU implementation of Projected Gradient Descent (PGD) with L2 ball projection.
pub fn step_pgd_cpu(param: &[f32], grad: &[f32], lr: f32, radius: f32, eps: f32) -> Vec<f32> {
    let d = param.len();
    let mut v = vec![0.0f64; d];
    let mut sum_sq = 0.0f64;
    for i in 0..d {
        let val = (param[i] as f64) - (lr as f64) * (grad[i] as f64);
        v[i] = val;
        sum_sq += val * val;
    }
    let norm_v = sum_sq.sqrt();
    let scale = if norm_v <= (radius as f64) {
        1.0
    } else {
        (radius as f64) / (norm_v + eps as f64)
    };

    v.iter().map(|&x| (x * scale) as f32).collect()
}

/// CPU implementation of Frank-Wolfe step on L1 ball ||s||_1 <= tau.
pub fn step_frank_wolfe_cpu(param: &[f32], grad: &[f32], tau: f32, gamma: f32) -> Vec<f32> {
    let d = param.len();
    let mut max_abs = -1.0f32;
    let mut max_idx = 0;
    for i in 0..d {
        let abs_g = grad[i].abs();
        if abs_g > max_abs {
            max_abs = abs_g;
            max_idx = i;
        }
    }

    let g_max = grad[max_idx];
    let sign_g = if g_max > 0.0 {
        1.0f32
    } else if g_max < 0.0 {
        -1.0f32
    } else {
        0.0f32
    };

    let mut out = vec![0.0f32; d];
    for i in 0..d {
        let s_i = if i == max_idx { -tau * sign_g } else { 0.0 };
        out[i] = (1.0 - gamma) * param[i] + gamma * s_i;
    }
    out
}

/// CPU implementation of Proximal Gradient (ISTA) with soft-thresholding.
pub fn step_ista_cpu(param: &[f32], grad: &[f32], lr: f32, lam: f32) -> Vec<f32> {
    let d = param.len();
    let thresh = lr * lam;
    let mut out = vec![0.0f32; d];
    for i in 0..d {
        let v = param[i] - lr * grad[i];
        let abs_v = v.abs();
        let sign_v = if v > 0.0 {
            1.0f32
        } else if v < 0.0 {
            -1.0f32
        } else {
            0.0f32
        };
        let shrunk = (abs_v - thresh).max(0.0);
        out[i] = sign_v * shrunk;
    }
    out
}

/// CPU implementation of Entropic Mirror Descent on probability simplex.
pub fn step_mirror_descent_cpu(param: &[f32], grad: &[f32], lr: f32, eps: f32) -> Vec<f32> {
    let d = param.len();
    let mut unnorm = vec![0.0f64; d];
    let mut sum_unnorm = 0.0f64;
    for i in 0..d {
        let u = (param[i] as f64) * (-(lr as f64) * (grad[i] as f64)).exp();
        unnorm[i] = u;
        sum_unnorm += u;
    }

    unnorm
        .iter()
        .map(|&u| (u / (sum_unnorm + eps as f64)) as f32)
        .collect()
}

/// Linear anchor forward pass: x @ w + b.
pub fn anchor_forward_cpu(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * out_dim];
    for r in 0..rows {
        for c in 0..out_dim {
            let mut sum = b[c];
            for k in 0..in_dim {
                sum += x[r * in_dim + k] * w[k * out_dim + c];
            }
            out[r * out_dim + c] = sum;
        }
    }
    out
}

/// Auto-dispatchers with CPU fallback.
pub fn step_pgd_auto(
    _device: &KernelDevice,
    param: &[f32],
    grad: &[f32],
    lr: f32,
    radius: f32,
    eps: f32,
) -> Vec<f32> {
    step_pgd_cpu(param, grad, lr, radius, eps)
}

pub fn step_frank_wolfe_auto(
    _device: &KernelDevice,
    param: &[f32],
    grad: &[f32],
    tau: f32,
    gamma: f32,
) -> Vec<f32> {
    step_frank_wolfe_cpu(param, grad, tau, gamma)
}

pub fn step_ista_auto(
    _device: &KernelDevice,
    param: &[f32],
    grad: &[f32],
    lr: f32,
    lam: f32,
) -> Vec<f32> {
    step_ista_cpu(param, grad, lr, lam)
}

pub fn step_mirror_descent_auto(
    _device: &KernelDevice,
    param: &[f32],
    grad: &[f32],
    lr: f32,
    eps: f32,
) -> Vec<f32> {
    step_mirror_descent_cpu(param, grad, lr, eps)
}
