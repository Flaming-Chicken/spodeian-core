//! Hardware Kernels (WGSL + Native CPU fallback) for Category 04 Topic 16:
//! 02_training_optimisations_progressive_stacking_data_pruning_mrl_and_plasticity in
//! 04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop.

use crate::context::KernelDevice;

/// WGSL Compute Shader for element-wise RHO-Loss computation.
pub const RHO_LOSS_SHADER: &str = r#"
struct RhoUniforms {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> uniforms: RhoUniforms;
@group(0) @binding(1) var<storage, read> loss_model: array<f32>;
@group(0) @binding(2) var<storage, read> loss_ref: array<f32>;
@group(0) @binding(3) var<storage, read_write> rho_out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.len) {
        return;
    }
    rho_out[idx] = loss_model[idx] - loss_ref[idx];
}
"#;

/// CPU implementation of MRL prefix projection.
/// z: [batch, d_in]
/// w: [d_in, d_out]
/// b: [d_out]
pub fn mrl_slice_forward_cpu(
    z: &[f32],
    w: &[f32],
    b: &[f32],
    batch: usize,
    d_in: usize,
    d_out: usize,
    slice_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * d_out];
    for i in 0..batch {
        for j in 0..d_out {
            let mut sum = b[j];
            for k in 0..slice_dim {
                sum += z[i * d_in + k] * w[k * d_out + j];
            }
            out[i * d_out + j] = sum;
        }
    }
    out
}

/// Auto-dispatcher for MRL prefix projection.
pub fn mrl_slice_forward_auto(
    _device: &KernelDevice,
    z: &[f32],
    w: &[f32],
    b: &[f32],
    batch: usize,
    d_in: usize,
    d_out: usize,
    slice_dim: usize,
) -> Vec<f32> {
    mrl_slice_forward_cpu(z, w, b, batch, d_in, d_out, slice_dim)
}

/// CPU implementation of EL2N sample importance scores.
pub fn compute_el2n_cpu(
    probs: &[f32],
    targets: &[f32],
    num_samples: usize,
    num_classes: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; num_samples];
    for i in 0..num_samples {
        let mut sum_sq = 0.0f32;
        let offset = i * num_classes;
        for j in 0..num_classes {
            let diff = probs[offset + j] - targets[offset + j];
            sum_sq += diff * diff;
        }
        out[i] = sum_sq.sqrt();
    }
    out
}

/// Auto-dispatcher for EL2N scores.
pub fn compute_el2n_auto(
    _device: &KernelDevice,
    probs: &[f32],
    targets: &[f32],
    num_samples: usize,
    num_classes: usize,
) -> Vec<f32> {
    compute_el2n_cpu(probs, targets, num_samples, num_classes)
}

/// CPU implementation of RHO-loss.
pub fn compute_rho_loss_cpu(loss_model: &[f32], loss_ref: &[f32]) -> Vec<f32> {
    let n = loss_model.len();
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        out[i] = loss_model[i] - loss_ref[i];
    }
    out
}

/// Auto-dispatcher for RHO-loss.
pub fn compute_rho_loss_auto(
    _device: &KernelDevice,
    loss_model: &[f32],
    loss_ref: &[f32],
) -> Vec<f32> {
    compute_rho_loss_cpu(loss_model, loss_ref)
}

/// CPU implementation of ReDo dormant neuron detection.
/// activations: [batch, hidden_dim]
pub fn detect_dormant_neurons_cpu(
    activations: &[f32],
    batch: usize,
    hidden_dim: usize,
    tau: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut mean_act = vec![0.0f32; hidden_dim];
    let inv_batch = 1.0f32 / (batch as f32);

    for j in 0..hidden_dim {
        let mut sum = 0.0f32;
        for i in 0..batch {
            sum += activations[i * hidden_dim + j].abs();
        }
        mean_act[j] = sum * inv_batch;
    }

    let mut dormant_mask = vec![0.0f32; hidden_dim];
    for j in 0..hidden_dim {
        if mean_act[j] < tau {
            dormant_mask[j] = 1.0f32;
        }
    }

    (mean_act, dormant_mask)
}

/// Auto-dispatcher for ReDo dormant neuron detection.
pub fn detect_dormant_neurons_auto(
    _device: &KernelDevice,
    activations: &[f32],
    batch: usize,
    hidden_dim: usize,
    tau: f32,
) -> (Vec<f32>, Vec<f32>) {
    detect_dormant_neurons_cpu(activations, batch, hidden_dim, tau)
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
