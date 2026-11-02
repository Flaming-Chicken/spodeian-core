//! Hardware Kernels (WGSL + Native CPU fallback) for Category 04 Topic 05:
//! 05_second_order_and_curvature_optimisers in 04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers.

use crate::context::KernelDevice;

/// WGSL Compute Shader for Sophia-G parameter update.
pub const SOPHIA_UPDATE_SHADER: &str = r#"
struct SophiaUniforms {
    len: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    weight_decay: f32,
    rho: f32,
    eps: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> uniforms: SophiaUniforms;
@group(0) @binding(1) var<storage, read> theta_in: array<f32>;
@group(0) @binding(2) var<storage, read> grad_in: array<f32>;
@group(0) @binding(3) var<storage, read> m_in: array<f32>;
@group(0) @binding(4) var<storage, read> h_in: array<f32>;
@group(0) @binding(5) var<storage, read> h_est_in: array<f32>;
@group(0) @binding(6) var<storage, read_write> theta_out: array<f32>;
@group(0) @binding(7) var<storage, read_write> m_out: array<f32>;
@group(0) @binding(8) var<storage, read_write> h_out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.len) {
        return;
    }

    let m_next = uniforms.beta1 * m_in[idx] + (1.0 - uniforms.beta1) * grad_in[idx];
    let h_next = uniforms.beta2 * h_in[idx] + (1.0 - uniforms.beta2) * h_est_in[idx];
    let denom = max(h_next, uniforms.eps);
    let ratio = m_next / denom;
    let clipped = clamp(ratio, -uniforms.rho, uniforms.rho);
    let th_next = theta_in[idx] * (1.0 - uniforms.lr * uniforms.weight_decay) - uniforms.lr * clipped;

    m_out[idx] = m_next;
    h_out[idx] = h_next;
    theta_out[idx] = th_next;
}
"#;

/// WGSL Compute Shader for AdaHessian parameter update.
pub const ADAHESSIAN_UPDATE_SHADER: &str = r#"
struct AdaHessianUniforms {
    len: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    weight_decay: f32,
    bias_m: f32,
    bias_v: f32,
    eps: f32,
};

@group(0) @binding(0) var<uniform> uniforms: AdaHessianUniforms;
@group(0) @binding(1) var<storage, read> theta_in: array<f32>;
@group(0) @binding(2) var<storage, read> grad_in: array<f32>;
@group(0) @binding(3) var<storage, read> m_in: array<f32>;
@group(0) @binding(4) var<storage, read> v_in: array<f32>;
@group(0) @binding(5) var<storage, read> d_in: array<f32>;
@group(0) @binding(6) var<storage, read_write> theta_out: array<f32>;
@group(0) @binding(7) var<storage, read_write> m_out: array<f32>;
@group(0) @binding(8) var<storage, read_write> v_out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.len) {
        return;
    }

    let m_next = uniforms.beta1 * m_in[idx] + (1.0 - uniforms.beta1) * grad_in[idx];
    let d_val = d_in[idx];
    let v_next = uniforms.beta2 * v_in[idx] + (1.0 - uniforms.beta2) * (d_val * d_val);

    let m_hat = m_next / uniforms.bias_m;
    let v_hat = v_next / uniforms.bias_v;
    let update = m_hat / (sqrt(v_hat) + uniforms.eps);
    let th_next = theta_in[idx] * (1.0 - uniforms.lr * uniforms.weight_decay) - uniforms.lr * update;

    m_out[idx] = m_next;
    v_out[idx] = v_next;
    theta_out[idx] = th_next;
}
"#;

/// CPU reference routine for L-BFGS two-loop recursion.
pub fn lbfgs_two_loop_cpu(s: &[f32], y: &[f32], grad: &[f32], m: usize, d: usize) -> Vec<f32> {
    let mut q = grad.to_vec();
    let mut alphas = vec![0.0f32; m];
    let mut rhos = vec![0.0f32; m];

    // Backward recursion
    for i in (0..m).rev() {
        let mut sy = 0.0f64;
        let mut sq = 0.0f64;
        for j in 0..d {
            let s_val = s[i * d + j] as f64;
            let y_val = y[i * d + j] as f64;
            let q_val = q[j] as f64;
            sy += s_val * y_val;
            sq += s_val * q_val;
        }
        let rho_i = (1.0 / sy) as f32;
        rhos[i] = rho_i;
        let alpha_i = (rho_i as f64 * sq) as f32;
        alphas[i] = alpha_i;
        for j in 0..d {
            q[j] -= alpha_i * y[i * d + j];
        }
    }

    // Initial scaling: gamma * I
    let mut s_last_y = 0.0f64;
    let mut y_last_y = 0.0f64;
    let last = m - 1;
    for j in 0..d {
        let s_val = s[last * d + j] as f64;
        let y_val = y[last * d + j] as f64;
        s_last_y += s_val * y_val;
        y_last_y += y_val * y_val;
    }
    let gamma = (s_last_y / y_last_y) as f32;
    let mut r = vec![0.0f32; d];
    for j in 0..d {
        r[j] = gamma * q[j];
    }

    // Forward recursion
    for i in 0..m {
        let mut yr = 0.0f64;
        for j in 0..d {
            yr += (y[i * d + j] as f64) * (r[j] as f64);
        }
        let beta = (rhos[i] as f64 * yr) as f32;
        for j in 0..d {
            r[j] += s[i * d + j] * (alphas[i] - beta);
        }
    }

    r
}

/// CPU reference routine for L-BFGS parameter update.
pub fn step_lbfgs_cpu(
    theta: &[f32],
    s: &[f32],
    y: &[f32],
    grad: &[f32],
    lr: f32,
    m: usize,
    d: usize,
) -> Vec<f32> {
    let dir = lbfgs_two_loop_cpu(s, y, grad, m, d);
    let mut out = vec![0.0f32; d];
    for j in 0..d {
        out[j] = theta[j] - lr * dir[j];
    }
    out
}

/// CPU reference routine for Sophia-G parameter update.
pub fn step_sophia_cpu(
    theta: &[f32],
    grad: &[f32],
    momentum: &[f32],
    hess_diag: &[f32],
    hess_diag_est: &[f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    weight_decay: f32,
    rho: f32,
    eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let d = theta.len();
    let mut th_out = vec![0.0f32; d];
    let mut m_out = vec![0.0f32; d];
    let mut h_out = vec![0.0f32; d];

    for j in 0..d {
        let m_val =
            (beta1 as f64 * (momentum[j] as f64) + (1.0 - beta1 as f64) * (grad[j] as f64)) as f32;
        let h_val = (beta2 as f64 * (hess_diag[j] as f64)
            + (1.0 - beta2 as f64) * (hess_diag_est[j] as f64)) as f32;
        let denom = (h_val as f64).max(eps as f64);
        let ratio = (m_val as f64) / denom;
        let clipped = ratio.clamp(-rho as f64, rho as f64);
        let th_val =
            (theta[j] as f64) * (1.0 - (lr as f64) * (weight_decay as f64)) - (lr as f64) * clipped;

        m_out[j] = m_val;
        h_out[j] = h_val;
        th_out[j] = th_val as f32;
    }

    (th_out, m_out, h_out)
}

/// CPU reference routine for AdaHessian parameter update.
pub fn step_adahessian_cpu(
    theta: &[f32],
    grad: &[f32],
    momentum: &[f32],
    v: &[f32],
    d_diag: &[f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    weight_decay: f32,
    step: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let d = theta.len();
    let bias_m = 1.0 - (beta1 as f64).powi(step as i32);
    let bias_v = 1.0 - (beta2 as f64).powi(step as i32);

    let mut th_out = vec![0.0f32; d];
    let mut m_out = vec![0.0f32; d];
    let mut v_out = vec![0.0f32; d];

    for j in 0..d {
        let m_val =
            (beta1 as f64 * (momentum[j] as f64) + (1.0 - beta1 as f64) * (grad[j] as f64)) as f32;
        let diag_val = d_diag[j] as f64;
        let v_val =
            (beta2 as f64 * (v[j] as f64) + (1.0 - beta2 as f64) * (diag_val * diag_val)) as f32;

        let m_hat = (m_val as f64) / bias_m;
        let v_hat = (v_val as f64) / bias_v;

        let update = m_hat / (v_hat.sqrt() + eps as f64);
        let th_val =
            (theta[j] as f64) * (1.0 - (lr as f64) * (weight_decay as f64)) - (lr as f64) * update;

        m_out[j] = m_val;
        v_out[j] = v_val;
        th_out[j] = th_val as f32;
    }

    (th_out, m_out, v_out)
}

/// CPU Cholesky linear solver for symmetric positive definite system A x = b in f64.
fn cholesky_solve_cpu(a: &[f64], b: &[f64], n: usize) -> Vec<f32> {
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                l[i * n + j] = sum.max(1e-12).sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }

    // Forward substitution
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i * n + k] * y[k];
        }
        y[i] = sum / l[i * n + i];
    }

    // Back substitution
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for k in (i + 1)..n {
            sum -= l[k * n + i] * x[k];
        }
        x[i] = sum / l[i * n + i];
    }

    x.into_iter().map(|v| v as f32).collect()
}

/// CPU reference routine for Levenberg-Marquardt parameter update.
pub fn step_lm_cpu(
    theta: &[f32],
    jacobian: &[f32],
    residual: &[f32],
    damping: f32,
    n_res: usize,
    d: usize,
) -> Vec<f32> {
    let mut a_mat = vec![0.0f64; d * d];
    for i in 0..d {
        for j in 0..d {
            let mut sum = 0.0f64;
            for k in 0..n_res {
                sum += (jacobian[k * d + i] as f64) * (jacobian[k * d + j] as f64);
            }
            if i == j {
                sum += damping as f64;
            }
            a_mat[i * d + j] = sum;
        }
    }

    let mut b_vec = vec![0.0f64; d];
    for i in 0..d {
        let mut sum = 0.0f64;
        for k in 0..n_res {
            sum += (jacobian[k * d + i] as f64) * (residual[k] as f64);
        }
        b_vec[i] = -sum;
    }

    let delta = cholesky_solve_cpu(&a_mat, &b_vec, d);
    let mut out = vec![0.0f32; d];
    for j in 0..d {
        out[j] = theta[j] + delta[j];
    }
    out
}

/// Auto-dispatchers with CPU fallback.
pub fn step_sophia_auto(
    _device: &KernelDevice,
    theta: &[f32],
    grad: &[f32],
    momentum: &[f32],
    hess_diag: &[f32],
    hess_diag_est: &[f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    weight_decay: f32,
    rho: f32,
    eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    step_sophia_cpu(
        theta,
        grad,
        momentum,
        hess_diag,
        hess_diag_est,
        lr,
        beta1,
        beta2,
        weight_decay,
        rho,
        eps,
    )
}

pub fn step_adahessian_auto(
    _device: &KernelDevice,
    theta: &[f32],
    grad: &[f32],
    momentum: &[f32],
    v: &[f32],
    d_diag: &[f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    weight_decay: f32,
    step: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    step_adahessian_cpu(
        theta,
        grad,
        momentum,
        v,
        d_diag,
        lr,
        beta1,
        beta2,
        weight_decay,
        step,
        eps,
    )
}

pub fn step_lbfgs_auto(
    _device: &KernelDevice,
    theta: &[f32],
    s: &[f32],
    y: &[f32],
    grad: &[f32],
    lr: f32,
    m: usize,
    d: usize,
) -> Vec<f32> {
    step_lbfgs_cpu(theta, s, y, grad, lr, m, d)
}

pub fn step_lm_auto(
    _device: &KernelDevice,
    theta: &[f32],
    jacobian: &[f32],
    residual: &[f32],
    damping: f32,
    n_res: usize,
    d: usize,
) -> Vec<f32> {
    step_lm_cpu(theta, jacobian, residual, damping, n_res, d)
}
