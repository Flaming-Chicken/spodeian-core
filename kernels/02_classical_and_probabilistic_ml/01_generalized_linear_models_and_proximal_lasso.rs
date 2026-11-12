//! Hardware Kernels (WGSL + Native CPU fallback) for Category 02 Topic 01:
//! 01_generalized_linear_models_and_proximal_lasso.

use crate::context::KernelDevice;

/// WGSL Compute Shader for elementwise Huber loss evaluation.
pub const HUBER_LOSS_SHADER: &str = r#"
struct HuberUniforms {
    count: u32,
    delta: f32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> uniforms: HuberUniforms;
@group(0) @binding(1) var<storage, read> residuals: array<f32>;
@group(0) @binding(2) var<storage, read_write> loss_out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.count) {
        return;
    }
    let r = abs(residuals[idx]);
    let quad = min(r, uniforms.delta);
    let lin = r - quad;
    loss_out[idx] = 0.5f * quad * quad + uniforms.delta * lin;
}
"#;

/// Helper: Solves A * x = b for symmetric positive-definite A (d x d) via Cholesky decomposition A = L L^T.
fn solve_cholesky_spd(a: &[f32], b: &[f32], d: usize) -> Vec<f32> {
    let mut l = vec![0.0f32; d * d];
    for i in 0..d {
        for j in 0..=i {
            let mut sum = 0.0f32;
            for k in 0..j {
                sum += l[i * d + k] * l[j * d + k];
            }
            if i == j {
                let val = a[i * d + i] - sum;
                l[i * d + j] = if val > 1e-12f32 { val.sqrt() } else { 1e-6f32 };
            } else {
                l[i * d + j] = (a[i * d + j] - sum) / l[j * d + j];
            }
        }
    }

    // Forward substitution: L y = b
    let mut y = vec![0.0f32; d];
    for i in 0..d {
        let mut sum = 0.0f32;
        for k in 0..i {
            sum += l[i * d + k] * y[k];
        }
        y[i] = (b[i] - sum) / l[i * d + i];
    }

    // Back substitution: L^T x = y
    let mut x = vec![0.0f32; d];
    for i in (0..d).rev() {
        let mut sum = 0.0f32;
        for k in (i + 1)..d {
            sum += l[k * d + i] * x[k];
        }
        x[i] = (y[i] - sum) / l[i * d + i];
    }

    x
}

/// Solves Ordinary Least Squares via normal equations X^T X w = X^T y.
pub fn solve_ols_cpu(x: &[f32], y: &[f32], n: usize, d: usize) -> (Vec<f32>, f32) {
    let mut xtx = vec![0.0f32; d * d];
    let mut xty = vec![0.0f32; d];

    for r in 0..n {
        let y_r = y[r];
        for i in 0..d {
            xty[i] += x[r * d + i] * y_r;
            for j in 0..d {
                xtx[i * d + j] += x[r * d + i] * x[r * d + j];
            }
        }
    }

    let w = solve_cholesky_spd(&xtx, &xty, d);

    let mut sum_sq = 0.0f32;
    for r in 0..n {
        let mut pred = 0.0f32;
        for j in 0..d {
            pred += x[r * d + j] * w[j];
        }
        let diff = pred - y[r];
        sum_sq += diff * diff;
    }
    let mse = sum_sq / (n as f32);

    (w, mse)
}

/// Auto-dispatcher for OLS solver.
pub fn solve_ols_auto(
    _device: &KernelDevice,
    x: &[f32],
    y: &[f32],
    n: usize,
    d: usize,
) -> (Vec<f32>, f32) {
    solve_ols_cpu(x, y, n, d)
}

/// Solves Ridge regression: (X^T X + lambda I) w = X^T y.
pub fn solve_ridge_cpu(x: &[f32], y: &[f32], n: usize, d: usize, lambda_reg: f32) -> Vec<f32> {
    let mut xtx = vec![0.0f32; d * d];
    let mut xty = vec![0.0f32; d];

    for r in 0..n {
        let y_r = y[r];
        for i in 0..d {
            xty[i] += x[r * d + i] * y_r;
            for j in 0..d {
                xtx[i * d + j] += x[r * d + i] * x[r * d + j];
            }
        }
    }

    for i in 0..d {
        xtx[i * d + i] += lambda_reg;
    }

    solve_cholesky_spd(&xtx, &xty, d)
}

/// Auto-dispatcher for Ridge regression.
pub fn solve_ridge_auto(
    _device: &KernelDevice,
    x: &[f32],
    y: &[f32],
    n: usize,
    d: usize,
    lambda_reg: f32,
) -> Vec<f32> {
    solve_ridge_cpu(x, y, n, d, lambda_reg)
}

/// Solves Lasso regression via FISTA (Beck & Teboulle, 2009).
pub fn solve_lasso_fista_cpu(
    x: &[f32],
    y: &[f32],
    n: usize,
    d: usize,
    lambda_reg: f32,
    steps: usize,
) -> Vec<f32> {
    let mut xtx = vec![0.0f32; d * d];
    for r in 0..n {
        for i in 0..d {
            for j in 0..d {
                xtx[i * d + j] += x[r * d + i] * x[r * d + j];
            }
        }
    }

    let mut v = vec![1.0f32; d];
    for _ in 0..20 {
        let mut next_v = vec![0.0f32; d];
        for i in 0..d {
            for j in 0..d {
                next_v[i] += xtx[i * d + j] * v[j];
            }
        }
        let norm: f32 = next_v.iter().map(|&val| val * val).sum::<f32>().sqrt();
        for i in 0..d {
            v[i] = next_v[i] / (norm + 1e-8f32);
        }
    }

    let mut eig_max = 0.0f32;
    for i in 0..d {
        let mut row_sum = 0.0f32;
        for j in 0..d {
            row_sum += xtx[i * d + j] * v[j];
        }
        eig_max += v[i] * row_sum;
    }

    let step_size = 1.0f32 / (eig_max + 1e-4f32);
    let mut w = vec![0.0f32; d];
    let mut y_k = vec![0.0f32; d];
    let mut t_k = 1.0f32;

    for _ in 0..steps {
        let mut pred = vec![0.0f32; n];
        for r in 0..n {
            for j in 0..d {
                pred[r] += x[r * d + j] * y_k[j];
            }
        }

        let mut grad = vec![0.0f32; d];
        for j in 0..d {
            for r in 0..n {
                grad[j] += x[r * d + j] * (pred[r] - y[r]);
            }
        }

        let mut w_next = vec![0.0f32; d];
        let threshold = step_size * lambda_reg;
        for j in 0..d {
            let z = y_k[j] - step_size * grad[j];
            if z > threshold {
                w_next[j] = z - threshold;
            } else if z < -threshold {
                w_next[j] = z + threshold;
            } else {
                w_next[j] = 0.0f32;
            }
        }

        let t_next = (1.0f32 + (1.0f32 + 4.0f32 * t_k * t_k).sqrt()) / 2.0f32;
        let factor = (t_k - 1.0f32) / t_next;
        for j in 0..d {
            y_k[j] = w_next[j] + factor * (w_next[j] - w[j]);
        }
        w = w_next;
        t_k = t_next;
    }

    w
}

/// Auto-dispatcher for Lasso regression.
pub fn solve_lasso_fista_auto(
    _device: &KernelDevice,
    x: &[f32],
    y: &[f32],
    n: usize,
    d: usize,
    lambda_reg: f32,
    steps: usize,
) -> Vec<f32> {
    solve_lasso_fista_cpu(x, y, n, d, lambda_reg, steps)
}

/// Poisson GLM step with log link.
pub fn step_poisson_glm_cpu(
    x: &[f32],
    y_counts: &[f32],
    w_init: &[f32],
    n: usize,
    d: usize,
    lr: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut mu = vec![0.0f32; n];
    for r in 0..n {
        let mut lin = 0.0f32;
        for j in 0..d {
            lin += x[r * d + j] * w_init[j];
        }
        mu[r] = lin.exp();
    }

    let mut grad = vec![0.0f32; d];
    for j in 0..d {
        for r in 0..n {
            grad[j] += x[r * d + j] * (mu[r] - y_counts[r]);
        }
    }

    let mut w_next = vec![0.0f32; d];
    for j in 0..d {
        w_next[j] = w_init[j] - lr * grad[j];
    }

    (grad, w_next)
}

/// Auto-dispatcher for Poisson GLM step.
pub fn step_poisson_glm_auto(
    _device: &KernelDevice,
    x: &[f32],
    y_counts: &[f32],
    w_init: &[f32],
    n: usize,
    d: usize,
    lr: f32,
) -> (Vec<f32>, Vec<f32>) {
    step_poisson_glm_cpu(x, y_counts, w_init, n, d, lr)
}

/// Mean Huber loss.
pub fn compute_huber_mean_loss_cpu(residual: &[f32], delta: f32) -> f32 {
    let mut sum_loss = 0.0f32;
    for &r in residual {
        let abs_r = r.abs();
        let quad = abs_r.min(delta);
        let lin = abs_r - quad;
        sum_loss += 0.5f32 * quad * quad + delta * lin;
    }
    sum_loss / (residual.len() as f32)
}

/// Auto-dispatcher for mean Huber loss.
pub fn compute_huber_mean_loss_auto(_device: &KernelDevice, residual: &[f32], delta: f32) -> f32 {
    compute_huber_mean_loss_cpu(residual, delta)
}

/// Forward pass for Linear Anchor module.
pub fn anchor_forward_cpu(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    batch: usize,
    in_features: usize,
    out_features: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * out_features];
    for i in 0..batch {
        for j in 0..out_features {
            let mut sum = b[j];
            for k in 0..in_features {
                sum += x[i * in_features + k] * w[k * out_features + j];
            }
            out[i * out_features + j] = sum;
        }
    }
    out
}
