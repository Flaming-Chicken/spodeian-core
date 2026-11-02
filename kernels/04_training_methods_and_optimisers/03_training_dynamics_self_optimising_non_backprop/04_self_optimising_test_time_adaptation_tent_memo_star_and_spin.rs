//! Hardware Kernels (WGSL + Native CPU fallback) for Category 04 Topic 18:
//! 04_self_optimising_test_time_adaptation_tent_memo_star_and_spin in
//! 04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop.

use crate::context::KernelDevice;

/// WGSL Compute Shader for MEMO marginal probability reduction across augmentations.
pub const MEMO_REDUCTION_SHADER: &str = r#"
struct MemoUniforms {
    k_aug: u32,
    num_classes: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> uniforms: MemoUniforms;
@group(0) @binding(1) var<storage, read> aug_probs: array<f32>;
@group(0) @binding(2) var<storage, read_write> marginal_p: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let c = global_id.x;
    if (c >= uniforms.num_classes) {
        return;
    }
    var sum_p = 0.0f;
    for (var k = 0u; k < uniforms.k_aug; k = k + 1u) {
        sum_p = sum_p + aug_probs[k * uniforms.num_classes + c];
    }
    marginal_p[c] = sum_p / f32(uniforms.k_aug);
}
"#;

/// CPU implementation of TENT step with analytical entropy gradients.
/// logits: [batch, num_classes], scale: [num_classes], shift: [num_classes]
pub fn compute_tent_step_cpu(
    logits: &[f32],
    scale: &[f32],
    shift: &[f32],
    batch: usize,
    num_classes: usize,
    lr: f32,
) -> (Vec<f32>, Vec<f32>, f32, Vec<f32>, Vec<f32>) {
    // 1. Compute softmax and log-softmax per sample
    let mut probs = vec![0.0f32; batch * num_classes];
    let mut log_probs = vec![0.0f32; batch * num_classes];
    let mut sample_entropies = vec![0.0f32; batch];
    let mut total_entropy = 0.0f32;

    for i in 0..batch {
        let offset = i * num_classes;
        let mut max_z = f32::NEG_INFINITY;
        for c in 0..num_classes {
            let z = logits[offset + c] * scale[c] + shift[c];
            if z > max_z {
                max_z = z;
            }
        }

        let mut sum_exp = 0.0f32;
        for c in 0..num_classes {
            let z = logits[offset + c] * scale[c] + shift[c];
            sum_exp += (z - max_z).exp();
        }
        let log_sum_exp = max_z + sum_exp.ln();

        let mut h_i = 0.0f32;
        for c in 0..num_classes {
            let z = logits[offset + c] * scale[c] + shift[c];
            let lp = z - log_sum_exp;
            let p = lp.exp();
            probs[offset + c] = p;
            log_probs[offset + c] = lp;
            h_i -= p * lp;
        }
        sample_entropies[i] = h_i;
        total_entropy += h_i;
    }
    let mean_entropy = total_entropy / (batch as f32);

    // 2. Compute analytical gradients: dH/dz_c = -p_c * (log p_c + H)
    let inv_b = 1.0f32 / (batch as f32);
    let mut grad_scale = vec![0.0f32; num_classes];
    let mut grad_shift = vec![0.0f32; num_classes];

    for i in 0..batch {
        let offset = i * num_classes;
        let h_i = sample_entropies[i];
        for c in 0..num_classes {
            let p = probs[offset + c];
            let lp = log_probs[offset + c];
            let dh_dz = -p * (lp + h_i);
            grad_scale[c] += inv_b * dh_dz * logits[offset + c];
            grad_shift[c] += inv_b * dh_dz;
        }
    }

    let mut scale_next = vec![0.0f32; num_classes];
    let mut shift_next = vec![0.0f32; num_classes];
    for c in 0..num_classes {
        scale_next[c] = scale[c] - lr * grad_scale[c];
        shift_next[c] = shift[c] - lr * grad_shift[c];
    }

    (scale_next, shift_next, mean_entropy, grad_scale, grad_shift)
}

/// Auto-dispatcher for TENT step.
pub fn compute_tent_step_auto(
    _device: &KernelDevice,
    logits: &[f32],
    scale: &[f32],
    shift: &[f32],
    batch: usize,
    num_classes: usize,
    lr: f32,
) -> (Vec<f32>, Vec<f32>, f32, Vec<f32>, Vec<f32>) {
    compute_tent_step_cpu(logits, scale, shift, batch, num_classes, lr)
}

/// CPU implementation of MEMO marginal entropy across augmentations.
pub fn compute_memo_marginal_entropy_cpu(
    aug_logits: &[f32],
    k_aug: usize,
    num_classes: usize,
    eps: f32,
) -> (Vec<f32>, f32) {
    let mut marginal_p = vec![0.0f32; num_classes];
    let inv_k = 1.0f32 / (k_aug as f32);

    for k in 0..k_aug {
        let offset = k * num_classes;
        let mut max_z = f32::NEG_INFINITY;
        for c in 0..num_classes {
            if aug_logits[offset + c] > max_z {
                max_z = aug_logits[offset + c];
            }
        }
        let mut sum_exp = 0.0f32;
        for c in 0..num_classes {
            sum_exp += (aug_logits[offset + c] - max_z).exp();
        }
        for c in 0..num_classes {
            let p = (aug_logits[offset + c] - max_z).exp() / sum_exp;
            marginal_p[c] += p * inv_k;
        }
    }

    let mut entropy = 0.0f32;
    for c in 0..num_classes {
        let p = marginal_p[c];
        entropy -= p * (p + eps).ln();
    }

    (marginal_p, entropy)
}

/// Auto-dispatcher for MEMO marginal entropy.
pub fn compute_memo_marginal_entropy_auto(
    _device: &KernelDevice,
    aug_logits: &[f32],
    k_aug: usize,
    num_classes: usize,
    eps: f32,
) -> (Vec<f32>, f32) {
    compute_memo_marginal_entropy_cpu(aug_logits, k_aug, num_classes, eps)
}

/// CPU implementation of SPIN self-play preference loss.
pub fn compute_spin_loss_cpu(logp_win: &[f32], logp_lose: &[f32], beta: f32) -> f32 {
    let n = logp_win.len();
    let mut sum_loss = 0.0f32;
    for i in 0..n {
        let diff = beta * (logp_win[i] - logp_lose[i]);
        // log_sigmoid(x) = -ln(1 + exp(-x))
        let log_sig = -((-diff).exp().ln_1p());
        sum_loss -= log_sig;
    }
    sum_loss / (n as f32)
}

/// Auto-dispatcher for SPIN loss.
pub fn compute_spin_loss_auto(
    _device: &KernelDevice,
    logp_win: &[f32],
    logp_lose: &[f32],
    beta: f32,
) -> f32 {
    compute_spin_loss_cpu(logp_win, logp_lose, beta)
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
