//! Hardware Kernels (WGSL + Native CPU fallback) for Category 04 Topic 19:
//! 05_non_backprop_forward_forward_predictive_coding_eqprop_dfa_hebbian in
//! 04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop.

use crate::context::KernelDevice;

/// WGSL Compute Shader for Forward-Forward layer normalization: h / (sqrt(sum(h^2)) + eps).
pub const FORWARD_FORWARD_NORM_SHADER: &str = r#"
struct FfUniforms {
    batch: u32,
    dim: u32,
    eps: f32,
    _pad0: u32,
};

@group(0) @binding(0) var<uniform> uniforms: FfUniforms;
@group(0) @binding(1) var<storage, read> h_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> h_norm_out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.x;
    if (b >= uniforms.batch) {
        return;
    }
    let offset = b * uniforms.dim;
    var sum_sq = 0.0f;
    for (var j = 0u; j < uniforms.dim; j = j + 1u) {
        let v = h_in[offset + j];
        sum_sq = sum_sq + v * v;
    }
    let norm = sqrt(sum_sq) + uniforms.eps;
    for (var j = 0u; j < uniforms.dim; j = j + 1u) {
        h_norm_out[offset + j] = h_in[offset + j] / norm;
    }
}
"#;

/// CPU implementation of Forward-Forward goodness loss and layer normalization.
pub fn forward_forward_loss_and_norm_cpu(
    h_pos: &[f32],
    h_neg: &[f32],
    batch: usize,
    dim: usize,
    theta: f32,
    eps: f32,
) -> (f32, Vec<f32>) {
    let mut total_loss = 0.0f32;
    let mut h_norm = vec![0.0f32; batch * dim];

    for i in 0..batch {
        let offset = i * dim;
        let mut g_p = 0.0f32;
        let mut g_n = 0.0f32;
        for j in 0..dim {
            let hp = h_pos[offset + j];
            let hn = h_neg[offset + j];
            g_p += hp * hp;
            g_n += hn * hn;
        }

        let loss_p = (theta - g_p).exp().ln_1p();
        let loss_n = (g_n - theta).exp().ln_1p();
        total_loss += loss_p + loss_n;

        let norm_p = g_p.sqrt() + eps;
        for j in 0..dim {
            h_norm[offset + j] = h_pos[offset + j] / norm_p;
        }
    }

    (total_loss / (batch as f32), h_norm)
}

/// Auto-dispatcher for Forward-Forward.
pub fn forward_forward_loss_and_norm_auto(
    _device: &KernelDevice,
    h_pos: &[f32],
    h_neg: &[f32],
    batch: usize,
    dim: usize,
    theta: f32,
    eps: f32,
) -> (f32, Vec<f32>) {
    forward_forward_loss_and_norm_cpu(h_pos, h_neg, batch, dim, theta, eps)
}

/// CPU implementation of Predictive Coding representation inference step.
/// x, r: [dim], w: [dim, dim]
pub fn predictive_coding_step_cpu(
    x: &[f32],
    r: &[f32],
    w: &[f32],
    dim: usize,
    eta: f32,
    lam: f32,
) -> Vec<f32> {
    // 1. Prediction: pred = r @ W
    let mut pred = vec![0.0f32; dim];
    for j in 0..dim {
        let mut sum = 0.0f32;
        for k in 0..dim {
            sum += r[k] * w[k * dim + j];
        }
        pred[j] = sum;
    }

    // 2. Error: err = x - pred
    let mut err = vec![0.0f32; dim];
    for j in 0..dim {
        err[j] = x[j] - pred[j];
    }

    // 3. Grad: err @ W^T
    let mut grad = vec![0.0f32; dim];
    for j in 0..dim {
        let mut sum = 0.0f32;
        for k in 0..dim {
            sum += err[k] * w[j * dim + k];
        }
        grad[j] = sum;
    }

    // 4. Update: r' = r + eta * (grad - lam * r)
    let mut r_next = vec![0.0f32; dim];
    for j in 0..dim {
        r_next[j] = r[j] + eta * (grad[j] - lam * r[j]);
    }
    r_next
}

/// Auto-dispatcher for Predictive Coding step.
pub fn predictive_coding_step_auto(
    _device: &KernelDevice,
    x: &[f32],
    r: &[f32],
    w: &[f32],
    dim: usize,
    eta: f32,
    lam: f32,
) -> Vec<f32> {
    predictive_coding_step_cpu(x, r, w, dim, eta, lam)
}

/// CPU implementation of Direct Feedback Alignment (DFA).
/// top_error: [batch, c], dfa_feedback: [dim, c], act_pre: [batch, dim]
pub fn dfa_project_error_cpu(
    top_error: &[f32],
    dfa_feedback: &[f32],
    act_pre: &[f32],
    batch: usize,
    dim: usize,
    c_out: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * dim];
    for i in 0..batch {
        for j in 0..dim {
            let mask = if act_pre[i * dim + j] > 0.0f32 {
                1.0f32
            } else {
                0.0f32
            };
            let mut sum = 0.0f32;
            for c in 0..c_out {
                sum += top_error[i * c_out + c] * dfa_feedback[j * c_out + c];
            }
            out[i * dim + j] = sum * mask;
        }
    }
    out
}

/// Auto-dispatcher for DFA projection.
pub fn dfa_project_error_auto(
    _device: &KernelDevice,
    top_error: &[f32],
    dfa_feedback: &[f32],
    act_pre: &[f32],
    batch: usize,
    dim: usize,
    c_out: usize,
) -> Vec<f32> {
    dfa_project_error_cpu(top_error, dfa_feedback, act_pre, batch, dim, c_out)
}

/// CPU implementation of Oja's Hebbian PCA update.
pub fn oja_hebbian_step_cpu(x: &[f32], w: &[f32], eta: f32) -> Vec<f32> {
    let dim = x.len();
    let mut y = 0.0f32;
    for j in 0..dim {
        y += x[j] * w[j];
    }
    let mut w_next = vec![0.0f32; dim];
    for j in 0..dim {
        w_next[j] = w[j] + eta * (y * x[j] - y * y * w[j]);
    }
    w_next
}

/// Auto-dispatcher for Oja's Hebbian update.
pub fn oja_hebbian_step_auto(_device: &KernelDevice, x: &[f32], w: &[f32], eta: f32) -> Vec<f32> {
    oja_hebbian_step_cpu(x, w, eta)
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
