//! Hardware Kernels (WGSL + Native CPU fallback) for Category 04 Topic 06:
//! 06_schedule_free_and_parameter_free_optimisers in 04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers.

use crate::context::KernelDevice;

/// WGSL Compute Shader for Schedule-Free AdamW parameter update.
pub const SCHEDULE_FREE_UPDATE_SHADER: &str = r#"
struct ScheduleFreeUniforms {
    len: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    weight_decay: f32,
    bias_v: f32,
    c_prime: f32,
    c_next: f32,
    eps: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> uniforms: ScheduleFreeUniforms;
@group(0) @binding(1) var<storage, read> x_in: array<f32>;
@group(0) @binding(2) var<storage, read> z_in: array<f32>;
@group(0) @binding(3) var<storage, read> y_in: array<f32>;
@group(0) @binding(4) var<storage, read> grad_in: array<f32>;
@group(0) @binding(5) var<storage, read> m_in: array<f32>;
@group(0) @binding(6) var<storage, read> v_in: array<f32>;
@group(0) @binding(7) var<storage, read_write> x_out: array<f32>;
@group(0) @binding(8) var<storage, read_write> z_out: array<f32>;
@group(0) @binding(9) var<storage, read_write> y_out: array<f32>;
@group(0) @binding(10) var<storage, read_write> m_out: array<f32>;
@group(0) @binding(11) var<storage, read_write> v_out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.len) {
        return;
    }

    let g = grad_in[idx];
    let m_next = uniforms.beta1 * m_in[idx] + (1.0 - uniforms.beta1) * g;
    let v_next = uniforms.beta2 * v_in[idx] + (1.0 - uniforms.beta2) * (g * g);
    let v_hat = v_next / uniforms.bias_v;

    let u = (m_next / (sqrt(v_hat) + uniforms.eps)) + uniforms.weight_decay * y_in[idx];
    let z_next = z_in[idx] - uniforms.lr * u;
    let x_next = (1.0 - uniforms.c_prime) * x_in[idx] + uniforms.c_prime * z_next;
    let y_next = (1.0 - uniforms.c_next) * z_next + uniforms.c_next * x_next;

    m_out[idx] = m_next;
    v_out[idx] = v_next;
    z_out[idx] = z_next;
    x_out[idx] = x_next;
    y_out[idx] = y_next;
}
"#;

/// CPU reference routine for Schedule-Free AdamW step.
pub fn step_schedule_free_cpu(
    x: &[f32],
    z: &[f32],
    y: &[f32],
    grad: &[f32],
    m: &[f32],
    v: &[f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    weight_decay: f32,
    step: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let d = x.len();
    let bias_v = 1.0 - (beta2 as f64).powi(step as i32);
    let c_prime = 1.0 / ((step + 1) as f64);
    let c_next = 1.0 / ((step + 2) as f64);

    let mut x_out = vec![0.0f32; d];
    let mut z_out = vec![0.0f32; d];
    let mut y_out = vec![0.0f32; d];
    let mut m_out = vec![0.0f32; d];
    let mut v_out = vec![0.0f32; d];

    for j in 0..d {
        let g = grad[j] as f64;
        let m_val = (beta1 as f64) * (m[j] as f64) + (1.0 - beta1 as f64) * g;
        let v_val = (beta2 as f64) * (v[j] as f64) + (1.0 - beta2 as f64) * (g * g);
        let v_hat = v_val / bias_v;

        let u = (m_val / (v_hat.sqrt() + eps as f64)) + (weight_decay as f64) * (y[j] as f64);
        let z_val = (z[j] as f64) - (lr as f64) * u;
        let x_val = (1.0 - c_prime) * (x[j] as f64) + c_prime * z_val;
        let y_val = (1.0 - c_next) * z_val + c_next * x_val;

        m_out[j] = m_val as f32;
        v_out[j] = v_val as f32;
        z_out[j] = z_val as f32;
        x_out[j] = x_val as f32;
        y_out[j] = y_val as f32;
    }

    (x_out, z_out, y_out, m_out, v_out)
}

/// CPU reference routine for Polyak's optimal step-size.
pub fn step_polyak_cpu(param: &[f32], grad: &[f32], f_val: f32, f_opt: f32, eps: f32) -> Vec<f32> {
    let d = param.len();
    let mut norm_sq = 0.0f64;
    for &g in grad {
        let g_f64 = g as f64;
        norm_sq += g_f64 * g_f64;
    }

    let gamma = ((f_val - f_opt) as f64) / (norm_sq + eps as f64);
    let mut out = vec![0.0f32; d];
    for j in 0..d {
        out[j] = ((param[j] as f64) - gamma * (grad[j] as f64)) as f32;
    }
    out
}

/// CPU reference routine for Barzilai-Borwein BB1 spectral step-size.
pub fn step_barzilai_borwein_cpu(
    param: &[f32],
    grad: &[f32],
    s: &[f32],
    y: &[f32],
    eps: f32,
) -> Vec<f32> {
    let d = param.len();
    let mut s_dot_s = 0.0f64;
    let mut s_dot_y = 0.0f64;
    for j in 0..d {
        let s_val = s[j] as f64;
        let y_val = y[j] as f64;
        s_dot_s += s_val * s_val;
        s_dot_y += s_val * y_val;
    }

    let gamma = s_dot_s / (s_dot_y + eps as f64);
    let mut out = vec![0.0f32; d];
    for j in 0..d {
        out[j] = ((param[j] as f64) - gamma * (grad[j] as f64)) as f32;
    }
    out
}

/// CPU reference routine for D-Adaptation step.
pub fn step_d_adaptation_cpu(
    param: &[f32],
    grad: &[f32],
    d_est: f32,
    lr0: f32,
    eps: f32,
) -> (Vec<f32>, f32) {
    let d = param.len();
    let mut l1_norm = 0.0f64;
    let mut l2_norm_sq = 0.0f64;
    for &g in grad {
        let abs_g = (g as f64).abs();
        l1_norm += abs_g;
        l2_norm_sq += abs_g * abs_g;
    }
    let l2_norm = l2_norm_sq.sqrt();

    let d_next = ((d_est as f64) + 0.05 * (l1_norm / ((d as f64).sqrt()) - 1.0)).max(0.1);
    let eff_lr = (lr0 as f64) * d_next;

    let mut out = vec![0.0f32; d];
    for j in 0..d {
        let step = eff_lr * (grad[j] as f64) / (l2_norm + eps as f64);
        out[j] = ((param[j] as f64) - step) as f32;
    }

    (out, d_next as f32)
}

/// Auto-dispatchers with CPU fallback.
pub fn step_schedule_free_auto(
    _device: &KernelDevice,
    x: &[f32],
    z: &[f32],
    y: &[f32],
    grad: &[f32],
    m: &[f32],
    v: &[f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    weight_decay: f32,
    step: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    step_schedule_free_cpu(
        x,
        z,
        y,
        grad,
        m,
        v,
        lr,
        beta1,
        beta2,
        weight_decay,
        step,
        eps,
    )
}

pub fn step_polyak_auto(
    _device: &KernelDevice,
    param: &[f32],
    grad: &[f32],
    f_val: f32,
    f_opt: f32,
    eps: f32,
) -> Vec<f32> {
    step_polyak_cpu(param, grad, f_val, f_opt, eps)
}

pub fn step_barzilai_borwein_auto(
    _device: &KernelDevice,
    param: &[f32],
    grad: &[f32],
    s: &[f32],
    y: &[f32],
    eps: f32,
) -> Vec<f32> {
    step_barzilai_borwein_cpu(param, grad, s, y, eps)
}

pub fn step_d_adaptation_auto(
    _device: &KernelDevice,
    param: &[f32],
    grad: &[f32],
    d_est: f32,
    lr0: f32,
    eps: f32,
) -> (Vec<f32>, f32) {
    step_d_adaptation_cpu(param, grad, d_est, lr0, eps)
}
