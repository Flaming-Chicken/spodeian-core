//! Hardware Kernels (WGSL + Native CPU fallback) for Category 04 Topic 21:
//! 07_scaling_laws_chinchilla_data_dedup_doremi_and_model_collapse in
//! 04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop.

use crate::context::KernelDevice;

/// WGSL Compute Shader for MinHash pairwise Jaccard similarity estimation.
pub const MINHASH_JACCARD_SHADER: &str = r#"
struct MinHashUniforms {
    b_docs: u32,
    k_hashes: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> uniforms: MinHashUniforms;
@group(0) @binding(1) var<storage, read> signatures: array<f32>;
@group(0) @binding(2) var<storage, read_write> jaccard_out: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;
    if (row >= uniforms.b_docs || col >= uniforms.b_docs) {
        return;
    }

    var matches = 0.0f;
    let k = uniforms.k_hashes;
    for (var idx = 0u; idx < k; idx = idx + 1u) {
        let val_i = signatures[row * k + idx];
        let val_j = signatures[col * k + idx];
        if (abs(val_i - val_j) < 0.0001f) {
            matches = matches + 1.0f;
        }
    }
    jaccard_out[row * uniforms.b_docs + col] = matches / f32(k);
}
"#;

/// CPU implementation of Chinchilla scaling loss and compute-optimal allocation.
pub fn compute_chinchilla_loss_and_opt_cpu(
    n: f32,
    d: f32,
    e_const: f32,
    a_const: f32,
    b_const: f32,
    alpha: f32,
    beta: f32,
    g_n: f32,
) -> (f32, f32, f32) {
    let loss = e_const + a_const * n.powf(-alpha) + b_const * d.powf(-beta);

    let exp_a = beta / (alpha + beta);
    let exp_b = alpha / (alpha + beta);
    let budget_c = 6.0f32 * n * d;
    let g_d = 1.0f32 / (6.0f32 * g_n);

    let n_opt = g_n * budget_c.powf(exp_a);
    let d_opt = g_d * budget_c.powf(exp_b);

    (loss, n_opt, d_opt)
}

/// Auto-dispatcher for Chinchilla scaling.
pub fn compute_chinchilla_loss_and_opt_auto(
    _device: &KernelDevice,
    n: f32,
    d: f32,
    e_const: f32,
    a_const: f32,
    b_const: f32,
    alpha: f32,
    beta: f32,
    g_n: f32,
) -> (f32, f32, f32) {
    compute_chinchilla_loss_and_opt_cpu(n, d, e_const, a_const, b_const, alpha, beta, g_n)
}

/// CPU implementation of DoReMi domain weight exponentiated gradient update.
pub fn doremi_update_step_cpu(
    w_init: &[f32],
    losses_model: &[f32],
    losses_ref: &[f32],
    eta: f32,
) -> (Vec<f32>, Vec<f32>) {
    let k = w_init.len();
    let mut excess_loss = vec![0.0f32; k];
    let mut unnorm = vec![0.0f32; k];
    let mut sum_exp = 0.0f32;

    for i in 0..k {
        let diff = losses_model[i] - losses_ref[i];
        excess_loss[i] = diff;
        let v = w_init[i].ln() + eta * diff;
        let ev = v.exp();
        unnorm[i] = ev;
        sum_exp += ev;
    }

    let mut w_next = vec![0.0f32; k];
    for i in 0..k {
        w_next[i] = unnorm[i] / sum_exp;
    }

    (excess_loss, w_next)
}

/// Auto-dispatcher for DoReMi step.
pub fn doremi_update_step_auto(
    _device: &KernelDevice,
    w_init: &[f32],
    losses_model: &[f32],
    losses_ref: &[f32],
    eta: f32,
) -> (Vec<f32>, Vec<f32>) {
    doremi_update_step_cpu(w_init, losses_model, losses_ref, eta)
}

/// CPU implementation of MinHash pairwise Jaccard similarity estimation.
pub fn minhash_pairwise_jaccard_cpu(
    signatures: &[f32],
    b_docs: usize,
    k_hashes: usize,
) -> Vec<f32> {
    let mut jaccard = vec![0.0f32; b_docs * b_docs];
    let k_f = k_hashes as f32;

    for i in 0..b_docs {
        for j in 0..b_docs {
            let mut matches = 0.0f32;
            for k in 0..k_hashes {
                let v_i = signatures[i * k_hashes + k];
                let v_j = signatures[j * k_hashes + k];
                if (v_i - v_j).abs() < 1e-6 {
                    matches += 1.0f32;
                }
            }
            jaccard[i * b_docs + j] = matches / k_f;
        }
    }
    jaccard
}

/// Auto-dispatcher for MinHash Jaccard matrix.
pub fn minhash_pairwise_jaccard_auto(
    _device: &KernelDevice,
    signatures: &[f32],
    b_docs: usize,
    k_hashes: usize,
) -> Vec<f32> {
    minhash_pairwise_jaccard_cpu(signatures, b_docs, k_hashes)
}

/// CPU implementation of Model Collapse variance accumulation cure.
pub fn model_collapse_variance_cure_cpu(var_synth: f32, var_real: f32, lambda_real: f32) -> f32 {
    (1.0f32 - lambda_real) * var_synth + lambda_real * var_real
}

/// Auto-dispatcher for Model Collapse variance cure.
pub fn model_collapse_variance_cure_auto(
    _device: &KernelDevice,
    var_synth: f32,
    var_real: f32,
    lambda_real: f32,
) -> f32 {
    model_collapse_variance_cure_cpu(var_synth, var_real, lambda_real)
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
