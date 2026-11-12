//! Hardware Kernels (WGSL + Native CPU fallback) for Category 04 Topic 13:
//! 06_sharpness_aware_swad_and_weight_averaging_dynamics in 04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation.

use crate::context::KernelDevice;

/// WGSL Compute Shader for Sharpness-Aware Minimization (SAM) step.
pub const SAM_STEP_SHADER: &str = r#"
struct SamUniforms {
    len: u32,
    rho: f32,
    lr: f32,
    weight_decay: f32,
    grad_norm: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> uniforms: SamUniforms;
@group(0) @binding(1) var<storage, read> param_in: array<f32>;
@group(0) @binding(2) var<storage, read> grad_at_w_in: array<f32>;
@group(0) @binding(3) var<storage, read> grad_at_pert_in: array<f32>;
@group(0) @binding(4) var<storage, read_write> param_out: array<f32>;
@group(0) @binding(5) var<storage, read_write> epsilon_out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.len) {
        return;
    }
    let p = param_in[idx];
    let gw = grad_at_w_in[idx];
    let gp = grad_at_pert_in[idx];

    let eps_i = uniforms.rho * (gw / uniforms.grad_norm);
    epsilon_out[idx] = eps_i;

    let update = gp + uniforms.weight_decay * p;
    param_out[idx] = p - uniforms.lr * update;
}
"#;

/// CPU implementation of SAM update step.
pub fn step_sam_cpu(
    param: &[f32],
    grad_at_w: &[f32],
    grad_at_pert: &[f32],
    rho: f32,
    lr: f32,
    weight_decay: f32,
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = param.len();
    let mut sum_sq = 0.0f64;
    for &g in grad_at_w {
        sum_sq += (g as f64) * (g as f64);
    }
    let grad_norm = (sum_sq.sqrt() as f32) + eps;
    let scale = rho / grad_norm;

    let mut out_p = vec![0.0f32; n];
    let mut out_eps = vec![0.0f32; n];

    for i in 0..n {
        let e = grad_at_w[i] * scale;
        out_eps[i] = e;
        let update = grad_at_pert[i] + weight_decay * param[i];
        out_p[i] = param[i] - lr * update;
    }

    (out_p, out_eps)
}

/// CPU implementation of Lookahead synchronization.
pub fn step_lookahead_cpu(
    slow_param: &[f32],
    fast_param: &[f32],
    alpha: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = slow_param.len();
    let mut next_s = vec![0.0f32; n];
    for i in 0..n {
        let s = slow_param[i];
        let f = fast_param[i];
        next_s[i] = s + alpha * (f - s);
    }
    let next_f = next_s.clone();
    (next_s, next_f)
}

/// CPU implementation of SWA and EMA.
pub fn step_swa_ema_cpu(
    swa_avg: &[f32],
    ema_avg: &[f32],
    curr_param: &[f32],
    n_models: usize,
    decay: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = swa_avg.len();
    let mut next_swa = vec![0.0f32; n];
    let mut next_ema = vec![0.0f32; n];
    let nm = n_models as f32;

    for i in 0..n {
        let s = swa_avg[i];
        let e = ema_avg[i];
        let c = curr_param[i];
        next_swa[i] = (s * nm + c) / (nm + 1.0);
        next_ema[i] = decay * e + (1.0 - decay) * c;
    }

    (next_swa, next_ema)
}

/// CPU implementation of Uniform Model Soup averaging across M models of length d.
pub fn uniform_model_soup_cpu(models: &[f32], num_models: usize, dim: usize) -> Vec<f32> {
    let mut soup = vec![0.0f32; dim];
    let inv_m = 1.0 / (num_models as f32);

    for m in 0..num_models {
        for d in 0..dim {
            soup[d] += models[m * dim + d] * inv_m;
        }
    }

    soup
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
pub fn step_sam_auto(
    _device: &KernelDevice,
    param: &[f32],
    grad_at_w: &[f32],
    grad_at_pert: &[f32],
    rho: f32,
    lr: f32,
    weight_decay: f32,
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    step_sam_cpu(param, grad_at_w, grad_at_pert, rho, lr, weight_decay, eps)
}

pub fn step_lookahead_auto(
    _device: &KernelDevice,
    slow_param: &[f32],
    fast_param: &[f32],
    alpha: f32,
) -> (Vec<f32>, Vec<f32>) {
    step_lookahead_cpu(slow_param, fast_param, alpha)
}

pub fn step_swa_ema_auto(
    _device: &KernelDevice,
    swa_avg: &[f32],
    ema_avg: &[f32],
    curr_param: &[f32],
    n_models: usize,
    decay: f32,
) -> (Vec<f32>, Vec<f32>) {
    step_swa_ema_cpu(swa_avg, ema_avg, curr_param, n_models, decay)
}

pub fn uniform_model_soup_auto(
    _device: &KernelDevice,
    models: &[f32],
    num_models: usize,
    dim: usize,
) -> Vec<f32> {
    uniform_model_soup_cpu(models, num_models, dim)
}
