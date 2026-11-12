//! Hardware Kernels (WGSL + Native CPU fallback) for Category 04 Topic 17:
//! 03_self_optimising_hypergradients_learned_optimisers_and_meta_reweighting in
//! 04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop.

use crate::context::KernelDevice;

/// WGSL Compute Shader for element-wise hypergradient descent parameter update.
pub const HYPERGRADIENT_DESCENT_SHADER: &str = r#"
struct HdUniforms {
    len: u32,
    lr: f32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> uniforms: HdUniforms;
@group(0) @binding(1) var<storage, read> w_in: array<f32>;
@group(0) @binding(2) var<storage, read> g_curr: array<f32>;
@group(0) @binding(3) var<storage, read_write> w_out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.len) {
        return;
    }
    w_out[idx] = w_in[idx] - uniforms.lr * g_curr[idx];
}
"#;

/// CPU implementation of Hypergradient Descent step.
pub fn step_hypergradient_descent_cpu(
    w: &[f32],
    g_curr: &[f32],
    g_prev: &[f32],
    lr: f32,
    hyper_lr: f32,
) -> (Vec<f32>, f32) {
    let d = w.len();
    let mut dot = 0.0f32;
    for i in 0..d {
        dot += g_curr[i] * g_prev[i];
    }
    let lr_next = lr + hyper_lr * dot;
    let mut w_next = vec![0.0f32; d];
    for i in 0..d {
        w_next[i] = w[i] - lr_next * g_curr[i];
    }
    (w_next, lr_next)
}

/// Auto-dispatcher for Hypergradient Descent.
pub fn step_hypergradient_descent_auto(
    _device: &KernelDevice,
    w: &[f32],
    g_curr: &[f32],
    g_prev: &[f32],
    lr: f32,
    hyper_lr: f32,
) -> (Vec<f32>, f32) {
    step_hypergradient_descent_cpu(w, g_curr, g_prev, lr, hyper_lr)
}

/// CPU implementation of Meta-Sample Reweighting.
/// per_sample_grads: [batch, dim]
/// val_grad: [dim]
pub fn meta_sample_reweight_cpu(
    per_sample_grads: &[f32],
    val_grad: &[f32],
    batch: usize,
    dim: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut alignments = vec![0.0f32; batch];
    let mut sum_v = 0.0f32;
    for i in 0..batch {
        let mut dot = 0.0f32;
        let offset = i * dim;
        for j in 0..dim {
            dot += per_sample_grads[offset + j] * val_grad[j];
        }
        let v = if dot > 0.0f32 { dot } else { 0.0f32 };
        alignments[i] = v;
        sum_v += v;
    }

    let denom = sum_v + eps;
    let mut norm_weights = vec![0.0f32; batch];
    for i in 0..batch {
        norm_weights[i] = alignments[i] / denom;
    }

    let mut reweighted_grad = vec![0.0f32; dim];
    for j in 0..dim {
        let mut sum = 0.0f32;
        for i in 0..batch {
            sum += norm_weights[i] * per_sample_grads[i * dim + j];
        }
        reweighted_grad[j] = sum;
    }

    (norm_weights, reweighted_grad)
}

/// Auto-dispatcher for Meta-Sample Reweighting.
pub fn meta_sample_reweight_auto(
    _device: &KernelDevice,
    per_sample_grads: &[f32],
    val_grad: &[f32],
    batch: usize,
    dim: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    meta_sample_reweight_cpu(per_sample_grads, val_grad, batch, dim, eps)
}

/// CPU implementation of Coordinate-wise Learned Optimizer MLP step.
/// param, grad, momentum: [dim]
/// w1: [2, 4], b1: [4]
/// w2: [4, 1], b2: [1]
pub fn step_learned_optimizer_cpu(
    param: &[f32],
    grad: &[f32],
    momentum: &[f32],
    w1: &[f32],
    b1: &[f32],
    w2: &[f32],
    b2: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let dim = param.len();
    let mut p_next = vec![0.0f32; dim];
    let mut m_next = vec![0.0f32; dim];
    let mut delta_w = vec![0.0f32; dim];

    for j in 0..dim {
        let g = grad[j];
        let m = momentum[j];

        // Layer 1: [g, m] @ w1 + b1 -> relu -> [4]
        let mut h = [0.0f32; 4];
        for k in 0..4 {
            let act = g * w1[0 * 4 + k] + m * w1[1 * 4 + k] + b1[k];
            h[k] = if act > 0.0f32 { act } else { 0.0f32 };
        }

        // Layer 2: h @ w2 + b2 -> [1]
        let mut dw = b2[0];
        for k in 0..4 {
            dw += h[k] * w2[k * 1 + 0];
        }

        delta_w[j] = dw;
        p_next[j] = param[j] - dw;
        m_next[j] = 0.9f32 * m + 0.1f32 * g;
    }

    (p_next, m_next, delta_w)
}

/// Auto-dispatcher for Learned Optimizer MLP.
pub fn step_learned_optimizer_auto(
    _device: &KernelDevice,
    param: &[f32],
    grad: &[f32],
    momentum: &[f32],
    w1: &[f32],
    b1: &[f32],
    w2: &[f32],
    b2: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    step_learned_optimizer_cpu(param, grad, momentum, w1, b1, w2, b2)
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
