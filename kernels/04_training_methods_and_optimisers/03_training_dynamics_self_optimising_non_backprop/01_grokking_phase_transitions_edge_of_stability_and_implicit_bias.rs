//! Hardware Kernels (WGSL + Native CPU fallback) for Category 04 Topic 15:
//! 01_grokking_phase_transitions_edge_of_stability_and_implicit_bias in
//! 04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop.

use crate::context::KernelDevice;

/// WGSL Compute Shader for Grokfast EMA filtering step.
pub const GROKFAST_STEP_SHADER: &str = r#"
struct GrokfastUniforms {
    len: u32,
    alpha: f32,
    lambda_gf: f32,
    _pad0: f32,
};

@group(0) @binding(0) var<uniform> uniforms: GrokfastUniforms;
@group(0) @binding(1) var<storage, read> g_raw: array<f32>;
@group(0) @binding(2) var<storage, read> mu_prev: array<f32>;
@group(0) @binding(3) var<storage, read_write> g_out: array<f32>;
@group(0) @binding(4) var<storage, read_write> mu_out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.len) {
        return;
    }
    let m = uniforms.alpha * mu_prev[idx] + (1.0 - uniforms.alpha) * g_raw[idx];
    mu_out[idx] = m;
    g_out[idx] = g_raw[idx] + uniforms.lambda_gf * m;
}
"#;

/// CPU implementation of Grokfast step.
pub fn step_grokfast_cpu(
    g_raw: &[f32],
    mu_prev: &[f32],
    alpha: f32,
    lambda_gf: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = g_raw.len();
    let mut g_out = vec![0.0f32; n];
    let mut mu_out = vec![0.0f32; n];

    for i in 0..n {
        let m = alpha * mu_prev[i] + (1.0 - alpha) * g_raw[i];
        mu_out[i] = m;
        g_out[i] = g_raw[i] + lambda_gf * m;
    }

    (g_out, mu_out)
}

/// Auto-dispatcher for Grokfast step.
pub fn step_grokfast_auto(
    _device: &KernelDevice,
    g_raw: &[f32],
    mu_prev: &[f32],
    alpha: f32,
    lambda_gf: f32,
) -> (Vec<f32>, Vec<f32>) {
    step_grokfast_cpu(g_raw, mu_prev, alpha, lambda_gf)
}

/// CPU implementation of Edge of Stability (EOS) cubic oscillator step.
pub fn step_edge_of_stability_cpu(x: &[f32], h: &[f32], gamma: &[f32], eta: f32) -> Vec<f32> {
    let n = x.len();
    let mut x_out = vec![0.0f32; n];

    for i in 0..n {
        let xi = x[i];
        let grad = h[i] * xi + gamma[i] * (xi * xi * xi);
        x_out[i] = xi - eta * grad;
    }

    x_out
}

/// Auto-dispatcher for Edge of Stability (EOS) step.
pub fn step_edge_of_stability_auto(
    _device: &KernelDevice,
    x: &[f32],
    h: &[f32],
    gamma: &[f32],
    eta: f32,
) -> Vec<f32> {
    step_edge_of_stability_cpu(x, h, gamma, eta)
}

/// CPU implementation of Implicit Bias logistic margin step on separable classification.
pub fn step_implicit_bias_cpu(w: &[f32], x_matrix: &[f32], y_labels: &[f32], eta: f32) -> Vec<f32> {
    let num_samples = y_labels.len();
    let num_features = w.len();
    let mut grad = vec![0.0f32; num_features];

    for i in 0..num_samples {
        let mut dot = 0.0f32;
        let offset = i * num_features;
        for j in 0..num_features {
            dot += x_matrix[offset + j] * w[j];
        }

        let margin = y_labels[i] * dot;
        let sig = 1.0f32 / (1.0f32 + margin.exp());
        let scalar = -y_labels[i] * sig;

        for j in 0..num_features {
            grad[j] += scalar * x_matrix[offset + j];
        }
    }

    let inv_n = 1.0f32 / (num_samples as f32);
    let mut w_next = vec![0.0f32; num_features];
    for j in 0..num_features {
        grad[j] *= inv_n;
        w_next[j] = w[j] - eta * grad[j];
    }

    w_next
}

/// Auto-dispatcher for Implicit Bias step.
pub fn step_implicit_bias_auto(
    _device: &KernelDevice,
    w: &[f32],
    x_matrix: &[f32],
    y_labels: &[f32],
    eta: f32,
) -> Vec<f32> {
    step_implicit_bias_cpu(w, x_matrix, y_labels, eta)
}

/// Forward pass for Linear Anchor module.
/// x: [batch, in_features]
/// w: [in_features, out_features]
/// b: [out_features]
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
