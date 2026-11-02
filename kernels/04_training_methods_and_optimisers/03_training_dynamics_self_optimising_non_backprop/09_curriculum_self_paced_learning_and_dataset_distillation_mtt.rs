//! Hardware Kernels (WGSL + Native CPU fallback) for Category 04 Topic 23:
//! 09_curriculum_self_paced_learning_and_dataset_distillation_mtt in
//! 04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop.

use crate::context::KernelDevice;

/// WGSL Compute Shader for Self-Paced Learning (SPL) sample weight calculation.
pub const SPL_WEIGHT_SHADER: &str = r#"
struct SplUniforms {
    count: u32,
    pace_lambda: f32,
    linear_mode: u32,
    _pad0: u32,
};

@group(0) @binding(0) var<uniform> uniforms: SplUniforms;
@group(0) @binding(1) var<storage, read> losses: array<f32>;
@group(0) @binding(2) var<storage, read_write> weights_out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.count) {
        return;
    }
    let l = losses[idx];
    if (l < uniforms.pace_lambda) {
        if (uniforms.linear_mode == 1u) {
            weights_out[idx] = 1.0f - l / uniforms.pace_lambda;
        } else {
            weights_out[idx] = 1.0f;
        }
    } else {
        weights_out[idx] = 0.0f;
    }
}
"#;

/// CPU implementation of Self-Paced Learning (SPL) weights and weighted losses.
pub fn compute_spl_weights_and_loss_cpu(
    losses: &[f32],
    pace_lambda: f32,
) -> (Vec<f32>, Vec<f32>, f32, f32) {
    let n = losses.len();
    let mut h_vec = vec![0.0f32; n];
    let mut lin_vec = vec![0.0f32; n];

    let mut sum_h = 0.0f32;
    let mut sum_h_l = 0.0f32;
    let mut sum_lin = 0.0f32;
    let mut sum_lin_l = 0.0f32;

    for i in 0..n {
        let l = losses[i];
        if l < pace_lambda {
            h_vec[i] = 1.0f32;
            sum_h += 1.0f32;
            sum_h_l += l;

            let w_lin = 1.0f32 - l / pace_lambda;
            lin_vec[i] = w_lin;
            sum_lin += w_lin;
            sum_lin_l += w_lin * l;
        }
    }

    let loss_h = sum_h_l / (sum_h + 1e-8f32);
    let loss_lin = sum_lin_l / (sum_lin + 1e-8f32);

    (h_vec, lin_vec, loss_h, loss_lin)
}

/// Auto-dispatcher for SPL weights and loss.
pub fn compute_spl_weights_and_loss_auto(
    _device: &KernelDevice,
    losses: &[f32],
    pace_lambda: f32,
) -> (Vec<f32>, Vec<f32>, f32, f32) {
    compute_spl_weights_and_loss_cpu(losses, pace_lambda)
}

/// CPU implementation of Competence-based curriculum and active sample mask.
pub fn compute_competence_and_curriculum_mask_cpu(
    difficulties: &[f32],
    step_t: f32,
    t_warmup: f32,
    c0: f32,
) -> (f32, Vec<f32>) {
    let comp_raw = c0 + (1.0f32 - c0) * (step_t / t_warmup);
    let comp = comp_raw.clamp(0.0f32, 1.0f32);

    let mut mask = vec![0.0f32; difficulties.len()];
    for i in 0..difficulties.len() {
        if difficulties[i] <= comp {
            mask[i] = 1.0f32;
        }
    }

    (comp, mask)
}

/// Auto-dispatcher for Competence-based curriculum and active sample mask.
pub fn compute_competence_and_curriculum_mask_auto(
    _device: &KernelDevice,
    difficulties: &[f32],
    step_t: f32,
    t_warmup: f32,
    c0: f32,
) -> (f32, Vec<f32>) {
    compute_competence_and_curriculum_mask_cpu(difficulties, step_t, t_warmup, c0)
}

/// CPU implementation of Gradient Matching loss (Zhao et al., 2021).
pub fn compute_gradient_matching_loss_cpu(grad_synth: &[f32], grad_real: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_s_sq = 0.0f32;
    let mut norm_r_sq = 0.0f32;

    for i in 0..grad_synth.len() {
        dot += grad_synth[i] * grad_real[i];
        norm_s_sq += grad_synth[i] * grad_synth[i];
        norm_r_sq += grad_real[i] * grad_real[i];
    }

    let cos_sim = dot / (norm_s_sq.sqrt() * norm_r_sq.sqrt() + 1e-8f32);
    1.0f32 - cos_sim
}

/// Auto-dispatcher for Gradient Matching loss.
pub fn compute_gradient_matching_loss_auto(
    _device: &KernelDevice,
    grad_synth: &[f32],
    grad_real: &[f32],
) -> f32 {
    compute_gradient_matching_loss_cpu(grad_synth, grad_real)
}

/// CPU implementation of Trajectory Matching loss (MTT - Cazenavette et al., 2022).
pub fn compute_trajectory_matching_loss_cpu(
    theta_start: &[f32],
    theta_student: &[f32],
    theta_expert: &[f32],
) -> f32 {
    let mut num = 0.0f32;
    let mut denom = 0.0f32;

    for i in 0..theta_start.len() {
        let diff_stud = theta_student[i] - theta_expert[i];
        let diff_start = theta_start[i] - theta_expert[i];
        num += diff_stud * diff_stud;
        denom += diff_start * diff_start;
    }

    num / (denom + 1e-8f32)
}

/// Auto-dispatcher for Trajectory Matching loss.
pub fn compute_trajectory_matching_loss_auto(
    _device: &KernelDevice,
    theta_start: &[f32],
    theta_student: &[f32],
    theta_expert: &[f32],
) -> f32 {
    compute_trajectory_matching_loss_cpu(theta_start, theta_student, theta_expert)
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
