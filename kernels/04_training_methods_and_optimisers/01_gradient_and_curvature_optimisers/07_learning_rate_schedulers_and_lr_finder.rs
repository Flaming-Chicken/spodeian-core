//! Hardware Kernels (WGSL + Native CPU fallback) for Category 04 Topic 07:
//! 07_learning_rate_schedulers_and_lr_finder in 04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers.

use crate::context::KernelDevice;
use std::f64::consts::PI;

/// WGSL Compute Shader for AutoClip gradient clipping and parameter step.
pub const AUTOCLIP_UPDATE_SHADER: &str = r#"
struct AutoClipUniforms {
    len: u32,
    lr: f32,
    clip_ratio: f32,
    _pad0: f32,
};

@group(0) @binding(0) var<uniform> uniforms: AutoClipUniforms;
@group(0) @binding(1) var<storage, read> param_in: array<f32>;
@group(0) @binding(2) var<storage, read> grad_in: array<f32>;
@group(0) @binding(3) var<storage, read_write> param_out: array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.len) {
        return;
    }
    let cg = grad_in[idx] * uniforms.clip_ratio;
    grad_out[idx] = cg;
    param_out[idx] = param_in[idx] - uniforms.lr * cg;
}
"#;

/// WGSL Compute Shader for scheduled SGD parameter update.
pub const SCHEDULED_SGD_UPDATE_SHADER: &str = r#"
struct SgdStepUniforms {
    len: u32,
    lr: f32,
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0) var<uniform> uniforms: SgdStepUniforms;
@group(0) @binding(1) var<storage, read> param_in: array<f32>;
@group(0) @binding(2) var<storage, read> grad_in: array<f32>;
@group(0) @binding(3) var<storage, read_write> param_out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.len) {
        return;
    }
    param_out[idx] = param_in[idx] - uniforms.lr * grad_in[idx];
}
"#;

/// CPU implementation of Cosine Annealing with Linear Warmup.
pub fn schedule_cosine_warmup_cpu(
    step: usize,
    total_steps: usize,
    warmup_steps: usize,
    eta_min: f32,
    eta_max: f32,
) -> f32 {
    if step <= warmup_steps {
        return eta_min + (step as f32 / warmup_steps.max(1) as f32) * (eta_max - eta_min);
    }
    let ratio = (step - warmup_steps) as f64 / (total_steps - warmup_steps).max(1) as f64;
    (eta_min as f64 + 0.5 * ((eta_max - eta_min) as f64) * (1.0 + (PI * ratio).cos())) as f32
}

/// CPU implementation of Warmup-Stable-Decay (WSD).
pub fn schedule_wsd_cpu(
    step: usize,
    total_steps: usize,
    warmup_steps: usize,
    decay_steps: usize,
    eta_min: f32,
    eta_max: f32,
) -> f32 {
    if step <= warmup_steps {
        return eta_min + (step as f32 / warmup_steps.max(1) as f32) * (eta_max - eta_min);
    }
    let decay_start = total_steps - decay_steps;
    if step <= decay_start {
        return eta_max;
    }
    let ratio = (step - decay_start) as f64 / decay_steps.max(1) as f64;
    (eta_min as f64 + 0.5 * ((eta_max - eta_min) as f64) * (1.0 + (PI * ratio).cos())) as f32
}

/// CPU implementation of OneCycleLR.
pub fn schedule_onecycle_cpu(
    step: usize,
    total_steps: usize,
    peak_steps: usize,
    eta_max: f32,
    div_factor: f32,
    final_div_factor: f32,
) -> f32 {
    let eta_init = eta_max / div_factor;
    let eta_min = eta_init / final_div_factor;
    if step <= peak_steps {
        let ratio = step as f64 / peak_steps.max(1) as f64;
        return (eta_init as f64 + 0.5 * ((eta_max - eta_init) as f64) * (1.0 - (PI * ratio).cos()))
            as f32;
    }
    let ratio = (step - peak_steps) as f64 / (total_steps - peak_steps).max(1) as f64;
    (eta_min as f64 + 0.5 * ((eta_max - eta_min) as f64) * (1.0 + (PI * ratio).cos())) as f32
}

/// Linear interpolation percentile matching numpy percentile method.
pub fn compute_percentile_cpu(data: &[f32], q: f32) -> f32 {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted[0];
    }
    let idx = (q as f64 / 100.0) * ((n - 1) as f64);
    let low = idx.floor() as usize;
    let high = idx.ceil() as usize;
    let frac = (idx - (low as f64)) as f32;
    if low >= n - 1 {
        sorted[n - 1]
    } else {
        sorted[low] + frac * (sorted[high] - sorted[low])
    }
}

/// CPU reference implementation for AutoClip step.
pub fn step_autoclip_cpu(
    param: &[f32],
    grad: &[f32],
    norm_history: &[f32],
    percentile: f32,
    lr: f32,
    eps: f32,
) -> (Vec<f32>, Vec<f32>, f32) {
    let d = param.len();
    let tau = compute_percentile_cpu(norm_history, percentile);
    let grad_norm = (grad.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>()).sqrt() as f32;
    let clip_ratio = if tau < (grad_norm + eps) {
        tau / (grad_norm + eps)
    } else {
        1.0
    };

    let mut clipped_grad = vec![0.0f32; d];
    let mut param_next = vec![0.0f32; d];

    for i in 0..d {
        let cg = grad[i] * clip_ratio;
        clipped_grad[i] = cg;
        param_next[i] = param[i] - lr * cg;
    }

    (param_next, clipped_grad, tau)
}

/// CPU parameter update step: param - lr * grad.
pub fn step_sgd_cpu(param: &[f32], grad: &[f32], lr: f32) -> Vec<f32> {
    param
        .iter()
        .zip(grad.iter())
        .map(|(&p, &g)| p - lr * g)
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
pub fn schedule_cosine_warmup_auto(
    _device: &KernelDevice,
    step: usize,
    total_steps: usize,
    warmup_steps: usize,
    eta_min: f32,
    eta_max: f32,
) -> f32 {
    schedule_cosine_warmup_cpu(step, total_steps, warmup_steps, eta_min, eta_max)
}

pub fn schedule_wsd_auto(
    _device: &KernelDevice,
    step: usize,
    total_steps: usize,
    warmup_steps: usize,
    decay_steps: usize,
    eta_min: f32,
    eta_max: f32,
) -> f32 {
    schedule_wsd_cpu(
        step,
        total_steps,
        warmup_steps,
        decay_steps,
        eta_min,
        eta_max,
    )
}

pub fn schedule_onecycle_auto(
    _device: &KernelDevice,
    step: usize,
    total_steps: usize,
    peak_steps: usize,
    eta_max: f32,
    div_factor: f32,
    final_div_factor: f32,
) -> f32 {
    schedule_onecycle_cpu(
        step,
        total_steps,
        peak_steps,
        eta_max,
        div_factor,
        final_div_factor,
    )
}

pub fn step_autoclip_auto(
    _device: &KernelDevice,
    param: &[f32],
    grad: &[f32],
    norm_history: &[f32],
    percentile: f32,
    lr: f32,
    eps: f32,
) -> (Vec<f32>, Vec<f32>, f32) {
    step_autoclip_cpu(param, grad, norm_history, percentile, lr, eps)
}

pub fn step_sgd_auto(_device: &KernelDevice, param: &[f32], grad: &[f32], lr: f32) -> Vec<f32> {
    step_sgd_cpu(param, grad, lr)
}
