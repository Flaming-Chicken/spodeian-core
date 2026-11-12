//! Hardware Kernels (WGSL + Native CPU fallback) for Category 04 Topic 22:
//! 08_meta_parameter_sensitivity_fanova_sobol_meta_landscapes_and_effective_lr in
//! 04_training_methods_and_optimisers/03_training_dynamics_self_optimising_non_backprop.

use crate::context::KernelDevice;

/// WGSL Compute Shader for Effective Learning Rate scaling: out = eta / (sum(w^2) + eps).
pub const EFFECTIVE_LR_SHADER: &str = r#"
struct LrUniforms {
    dim: u32,
    eta_nominal: f32,
    eps: f32,
    _pad0: u32,
};

@group(0) @binding(0) var<uniform> uniforms: LrUniforms;
@group(0) @binding(1) var<storage, read> w_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> eff_lr_out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x > 0u) {
        return;
    }
    var sum_sq = 0.0f;
    for (var i = 0u; i < uniforms.dim; i = i + 1u) {
        let v = w_in[i];
        sum_sq = sum_sq + v * v;
    }
    eff_lr_out[0] = uniforms.eta_nominal / (sum_sq + uniforms.eps);
}
"#;

/// Helper for population variance (unbiased = false) of flat slices.
fn variance_slice(values: &[f32]) -> f32 {
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let mut sum_sq = 0.0f32;
    for &v in values {
        let diff = v - mean;
        sum_sq += diff * diff;
    }
    sum_sq / n
}

/// CPU implementation of fANOVA variance decomposition and Sobol' indices.
pub fn compute_fanova_sobol_indices_cpu(
    y_grid: &[f32],
    m: usize,
    n: usize,
) -> (f32, f32, f32, f32, f32, f32, f32) {
    let total_var = variance_slice(y_grid);

    // E[Y | x1]: mean across row i
    let mut e_given_x1 = vec![0.0f32; m];
    for i in 0..m {
        let mut sum = 0.0f32;
        for j in 0..n {
            sum += y_grid[i * n + j];
        }
        e_given_x1[i] = sum / (n as f32);
    }
    let v1 = variance_slice(&e_given_x1);

    // E[Y | x2]: mean across col j
    let mut e_given_x2 = vec![0.0f32; n];
    for j in 0..n {
        let mut sum = 0.0f32;
        for i in 0..m {
            sum += y_grid[i * n + j];
        }
        e_given_x2[j] = sum / (m as f32);
    }
    let v2 = variance_slice(&e_given_x2);

    let v12 = total_var - (v1 + v2);
    let s1 = v1 / total_var;
    let s2 = v2 / total_var;
    let st1 = (v1 + v12) / total_var;
    let st2 = (v2 + v12) / total_var;

    (total_var, v1, v2, s1, s2, st1, st2)
}

/// Auto-dispatcher for fANOVA / Sobol' indices.
pub fn compute_fanova_sobol_indices_auto(
    _device: &KernelDevice,
    y_grid: &[f32],
    m: usize,
    n: usize,
) -> (f32, f32, f32, f32, f32, f32, f32) {
    compute_fanova_sobol_indices_cpu(y_grid, m, n)
}

/// CPU implementation of Effective Learning Rate and Equilibrium Norm.
pub fn compute_effective_lr_and_eq_norm_cpu(
    w: &[f32],
    eta_nominal: f32,
    weight_decay: f32,
    grad_norm: f32,
) -> (f32, f32) {
    let mut sum_sq = 0.0f32;
    for &v in w {
        sum_sq += v * v;
    }
    let eff_lr = eta_nominal / sum_sq;
    let eq_norm = ((eta_nominal * grad_norm * grad_norm) / (2.0f32 * weight_decay)).powf(0.25f32);
    (eff_lr, eq_norm)
}

/// Auto-dispatcher for Effective Learning Rate and Equilibrium Norm.
pub fn compute_effective_lr_and_eq_norm_auto(
    _device: &KernelDevice,
    w: &[f32],
    eta_nominal: f32,
    weight_decay: f32,
    grad_norm: f32,
) -> (f32, f32) {
    compute_effective_lr_and_eq_norm_cpu(w, eta_nominal, weight_decay, grad_norm)
}

/// CPU implementation of Chaotic Meta-Gradient Explosion.
pub fn compute_chaotic_meta_gradient_explosion_cpu(
    eig_max: f32,
    eta_meta: f32,
    steps_t: f32,
) -> (f32, f32) {
    let mult = (1.0f32 - eta_meta * eig_max).abs();
    let accum = mult.powf(steps_t);
    (mult, accum)
}

/// Auto-dispatcher for Chaotic Meta-Gradient Explosion.
pub fn compute_chaotic_meta_gradient_explosion_auto(
    _device: &KernelDevice,
    eig_max: f32,
    eta_meta: f32,
    steps_t: f32,
) -> (f32, f32) {
    compute_chaotic_meta_gradient_explosion_cpu(eig_max, eta_meta, steps_t)
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
