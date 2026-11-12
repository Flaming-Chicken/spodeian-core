//! Hardware Kernels (WGSL + Native CPU fallback) for Category 04 Topic 09:
//! 02_multi_objective_pareto_mgda_tchebycheff_and_gradient_surgery in 04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation.

use crate::context::KernelDevice;

/// WGSL Compute Shader for convex combination gradient step (MGDA).
pub const MGDA_COMBINE_SHADER: &str = r#"
struct MgdaUniforms {
    len: u32,
    lr: f32,
    alpha: f32,
    _pad0: f32,
};

@group(0) @binding(0) var<uniform> uniforms: MgdaUniforms;
@group(0) @binding(1) var<storage, read> param_in: array<f32>;
@group(0) @binding(2) var<storage, read> g1_in: array<f32>;
@group(0) @binding(3) var<storage, read> g2_in: array<f32>;
@group(0) @binding(4) var<storage, read_write> param_out: array<f32>;
@group(0) @binding(5) var<storage, read_write> g_out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.len) {
        return;
    }
    let gm = uniforms.alpha * g1_in[idx] + (1.0 - uniforms.alpha) * g2_in[idx];
    g_out[idx] = gm;
    param_out[idx] = param_in[idx] - uniforms.lr * gm;
}
"#;

/// WGSL Compute Shader for linear multi-task scalarization.
pub const LINEAR_SCALARIZATION_SHADER: &str = r#"
struct LinUniforms {
    len: u32,
    lr: f32,
    w1: f32,
    w2: f32,
};

@group(0) @binding(0) var<uniform> uniforms: LinUniforms;
@group(0) @binding(1) var<storage, read> param_in: array<f32>;
@group(0) @binding(2) var<storage, read> g1_in: array<f32>;
@group(0) @binding(3) var<storage, read> g2_in: array<f32>;
@group(0) @binding(4) var<storage, read_write> param_out: array<f32>;
@group(0) @binding(5) var<storage, read_write> g_out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.len) {
        return;
    }
    let gl = uniforms.w1 * g1_in[idx] + uniforms.w2 * g2_in[idx];
    g_out[idx] = gl;
    param_out[idx] = param_in[idx] - uniforms.lr * gl;
}
"#;

/// CPU implementation of 2-task MGDA step.
pub fn step_mgda_2task_cpu(
    param: &[f32],
    g1: &[f32],
    g2: &[f32],
    lr: f32,
    eps: f32,
) -> (Vec<f32>, Vec<f32>, f32) {
    let d = param.len();
    let mut u_norm_sq = 0.0f64;
    let mut num = 0.0f64;
    for i in 0..d {
        let u_i = (g1[i] as f64) - (g2[i] as f64);
        u_norm_sq += u_i * u_i;
        num += ((g2[i] as f64) - (g1[i] as f64)) * (g2[i] as f64);
    }

    let alpha = (num / (u_norm_sq + eps as f64)).clamp(0.0, 1.0) as f32;

    let mut g_mgda = vec![0.0f32; d];
    let mut p_next = vec![0.0f32; d];
    for i in 0..d {
        let gm = alpha * g1[i] + (1.0 - alpha) * g2[i];
        g_mgda[i] = gm;
        p_next[i] = param[i] - lr * gm;
    }

    (p_next, g_mgda, alpha)
}

/// CPU implementation of PCGrad conflicting gradient projection step.
pub fn step_pcgrad_2task_cpu(
    param: &[f32],
    g1: &[f32],
    g2: &[f32],
    lr: f32,
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let d = param.len();
    let mut dot_12 = 0.0f64;
    let mut norm_sq_1 = 0.0f64;
    let mut norm_sq_2 = 0.0f64;
    for i in 0..d {
        let a = g1[i] as f64;
        let b = g2[i] as f64;
        dot_12 += a * b;
        norm_sq_1 += a * a;
        norm_sq_2 += b * b;
    }

    let mut g_pc = vec![0.0f32; d];
    let mut p_next = vec![0.0f32; d];

    if dot_12 < 0.0 {
        let scale_1 = dot_12 / (norm_sq_2 + eps as f64);
        let scale_2 = dot_12 / (norm_sq_1 + eps as f64);
        for i in 0..d {
            let g1_proj = (g1[i] as f64) - scale_1 * (g2[i] as f64);
            let g2_proj = (g2[i] as f64) - scale_2 * (g1[i] as f64);
            let gp = (0.5 * (g1_proj + g2_proj)) as f32;
            g_pc[i] = gp;
            p_next[i] = param[i] - lr * gp;
        }
    } else {
        for i in 0..d {
            let gp = 0.5 * (g1[i] + g2[i]);
            g_pc[i] = gp;
            p_next[i] = param[i] - lr * gp;
        }
    }

    (p_next, g_pc)
}

/// CPU implementation of Augmented Tchebycheff scalarization step.
pub fn step_tchebycheff_cpu(
    param: &[f32],
    g1: &[f32],
    g2: &[f32],
    losses: &[f32],
    z_star: &[f32],
    weights: &[f32],
    rho: f32,
    lr: f32,
) -> (Vec<f32>, Vec<f32>) {
    let d = param.len();
    let diff0 = weights[0] * (losses[0] - z_star[0]);
    let diff1 = weights[1] * (losses[1] - z_star[1]);

    let idx_max = if diff0 >= diff1 { 0 } else { 1 };

    let mut ds_dl = [rho * weights[0], rho * weights[1]];
    ds_dl[idx_max] += weights[idx_max];

    let mut g_tch = vec![0.0f32; d];
    let mut p_next = vec![0.0f32; d];

    for i in 0..d {
        let gt = ds_dl[0] * g1[i] + ds_dl[1] * g2[i];
        g_tch[i] = gt;
        p_next[i] = param[i] - lr * gt;
    }

    (p_next, g_tch)
}

/// CPU implementation of linear multi-task scalarization step.
pub fn step_linear_scalarization_cpu(
    param: &[f32],
    g1: &[f32],
    g2: &[f32],
    weights: &[f32],
    lr: f32,
) -> (Vec<f32>, Vec<f32>) {
    let d = param.len();
    let mut g_lin = vec![0.0f32; d];
    let mut p_next = vec![0.0f32; d];

    for i in 0..d {
        let gl = weights[0] * g1[i] + weights[1] * g2[i];
        g_lin[i] = gl;
        p_next[i] = param[i] - lr * gl;
    }

    (p_next, g_lin)
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
pub fn step_mgda_2task_auto(
    _device: &KernelDevice,
    param: &[f32],
    g1: &[f32],
    g2: &[f32],
    lr: f32,
    eps: f32,
) -> (Vec<f32>, Vec<f32>, f32) {
    step_mgda_2task_cpu(param, g1, g2, lr, eps)
}

pub fn step_pcgrad_2task_auto(
    _device: &KernelDevice,
    param: &[f32],
    g1: &[f32],
    g2: &[f32],
    lr: f32,
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    step_pcgrad_2task_cpu(param, g1, g2, lr, eps)
}

pub fn step_tchebycheff_auto(
    _device: &KernelDevice,
    param: &[f32],
    g1: &[f32],
    g2: &[f32],
    losses: &[f32],
    z_star: &[f32],
    weights: &[f32],
    rho: f32,
    lr: f32,
) -> (Vec<f32>, Vec<f32>) {
    step_tchebycheff_cpu(param, g1, g2, losses, z_star, weights, rho, lr)
}

pub fn step_linear_scalarization_auto(
    _device: &KernelDevice,
    param: &[f32],
    g1: &[f32],
    g2: &[f32],
    weights: &[f32],
    lr: f32,
) -> (Vec<f32>, Vec<f32>) {
    step_linear_scalarization_cpu(param, g1, g2, weights, lr)
}
