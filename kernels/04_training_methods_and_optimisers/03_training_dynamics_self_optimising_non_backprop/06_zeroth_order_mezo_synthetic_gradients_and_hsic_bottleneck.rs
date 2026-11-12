//! Hardware Kernels (WGSL + Native CPU fallback) for Category 04 Topic 20:
//! 06_zeroth_order_mezo_synthetic_gradients_and_hsic_bottleneck in
//! 04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop.

use crate::context::KernelDevice;

/// WGSL Compute Shader for MeZO parameter update: theta' = theta - eta * dir_deriv * z.
pub const MEZO_UPDATE_SHADER: &str = r#"
struct MezoUniforms {
    dim: u32,
    eta: f32,
    dir_deriv: f32,
    _pad0: u32,
};

@group(0) @binding(0) var<uniform> uniforms: MezoUniforms;
@group(0) @binding(1) var<storage, read> theta_in: array<f32>;
@group(0) @binding(2) var<storage, read> z_in: array<f32>;
@group(0) @binding(3) var<storage, read_write> theta_out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.dim) {
        return;
    }
    theta_out[idx] = theta_in[idx] - uniforms.eta * uniforms.dir_deriv * z_in[idx];
}
"#;

/// CPU implementation of MeZO directional derivative and parameter update.
pub fn mezo_step_cpu(
    theta: &[f32],
    z: &[f32],
    theta_star: &[f32],
    eps: f32,
    eta: f32,
) -> (f32, f32, f32, Vec<f32>) {
    let dim = theta.len();
    let mut diff_pos_sq = 0.0f32;
    let mut diff_neg_sq = 0.0f32;

    for i in 0..dim {
        let th_pos = theta[i] + eps * z[i];
        let th_neg = theta[i] - eps * z[i];
        let dp = th_pos - theta_star[i];
        let dn = th_neg - theta_star[i];
        diff_pos_sq += dp * dp;
        diff_neg_sq += dn * dn;
    }

    let loss_pos = 0.5f32 * diff_pos_sq;
    let loss_neg = 0.5f32 * diff_neg_sq;
    let dir_deriv = (loss_pos - loss_neg) / (2.0f32 * eps);

    let mut theta_next = vec![0.0f32; dim];
    for i in 0..dim {
        theta_next[i] = theta[i] - eta * dir_deriv * z[i];
    }

    (loss_pos, loss_neg, dir_deriv, theta_next)
}

/// Auto-dispatcher for MeZO step.
pub fn mezo_step_auto(
    _device: &KernelDevice,
    theta: &[f32],
    z: &[f32],
    theta_star: &[f32],
    eps: f32,
    eta: f32,
) -> (f32, f32, f32, Vec<f32>) {
    mezo_step_cpu(theta, z, theta_star, eps, eta)
}

/// CPU implementation of Decoupled Neural Interfaces (DNI) synthetic gradient.
pub fn dni_synthetic_gradient_cpu(
    h: &[f32],
    w_synth: &[f32],
    b_synth: &[f32],
    target_grad: &[f32],
    batch: usize,
    dim: usize,
) -> (Vec<f32>, f32) {
    let mut pred_grad = vec![0.0f32; batch * dim];
    let mut total_diff_sq = 0.0f32;

    for i in 0..batch {
        for j in 0..dim {
            let mut sum = b_synth[j];
            for k in 0..dim {
                sum += h[i * dim + k] * w_synth[k * dim + j];
            }
            pred_grad[i * dim + j] = sum;
            let diff = sum - target_grad[i * dim + j];
            total_diff_sq += diff * diff;
        }
    }

    let dni_loss = 0.5f32 * (total_diff_sq / ((batch * dim) as f32));
    (pred_grad, dni_loss)
}

/// Auto-dispatcher for DNI synthetic gradient.
pub fn dni_synthetic_gradient_auto(
    _device: &KernelDevice,
    h: &[f32],
    w_synth: &[f32],
    b_synth: &[f32],
    target_grad: &[f32],
    batch: usize,
    dim: usize,
) -> (Vec<f32>, f32) {
    dni_synthetic_gradient_cpu(h, w_synth, b_synth, target_grad, batch, dim)
}

/// CPU implementation of Hilbert-Schmidt Independence Criterion (HSIC).
pub fn hsic_bottleneck_cpu(h: &[f32], y: &[f32], batch: usize, dim_h: usize, dim_y: usize) -> f32 {
    let mut k_mat = vec![0.0f32; batch * batch];
    let mut l_mat = vec![0.0f32; batch * batch];

    // Compute linear kernels K = H H^T and L = Y Y^T
    for i in 0..batch {
        for j in 0..batch {
            let mut sum_k = 0.0f32;
            for d in 0..dim_h {
                sum_k += h[i * dim_h + d] * h[j * dim_h + d];
            }
            k_mat[i * batch + j] = sum_k;

            let mut sum_l = 0.0f32;
            for d in 0..dim_y {
                sum_l += y[i * dim_y + d] * y[j * dim_y + d];
            }
            l_mat[i * batch + j] = sum_l;
        }
    }

    // Centering H_c = I - 1/B * 1 1^T
    // K_c = H_c K H_c
    let b_f = batch as f32;
    let mut h_c = vec![0.0f32; batch * batch];
    for i in 0..batch {
        for j in 0..batch {
            let delta = if i == j { 1.0f32 } else { 0.0f32 };
            h_c[i * batch + j] = delta - 1.0f32 / b_f;
        }
    }

    // Helper to multiply [B, B] by [B, B]
    let matmul_bb = |a: &[f32], b: &[f32]| -> Vec<f32> {
        let mut out = vec![0.0f32; batch * batch];
        for i in 0..batch {
            for j in 0..batch {
                let mut sum = 0.0f32;
                for k in 0..batch {
                    sum += a[i * batch + k] * b[k * batch + j];
                }
                out[i * batch + j] = sum;
            }
        }
        out
    };

    let k_h = matmul_bb(&h_c, &k_mat);
    let k_c = matmul_bb(&k_h, &h_c);

    let l_h = matmul_bb(&h_c, &l_mat);
    let l_c = matmul_bb(&l_h, &h_c);

    // Tr(K_c L_c) = sum(K_c * L_c)
    let mut tr = 0.0f32;
    for i in 0..(batch * batch) {
        tr += k_c[i] * l_c[i];
    }

    let denom = (b_f - 1.0f32) * (b_f - 1.0f32);
    tr / denom
}

/// Auto-dispatcher for HSIC bottleneck.
pub fn hsic_bottleneck_auto(
    _device: &KernelDevice,
    h: &[f32],
    y: &[f32],
    batch: usize,
    dim_h: usize,
    dim_y: usize,
) -> f32 {
    hsic_bottleneck_cpu(h, y, batch, dim_h, dim_y)
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
