//! Hardware Kernels (WGSL + Native CPU fallback) for Category 04 Topic 12:
//! 05_multi_fidelity_hyperparameter_optimisation_hyperband_asha_and_boed in 04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation.

use crate::context::KernelDevice;

/// WGSL Compute Shader for 1D Gaussian kernel log-evaluation in TPE.
pub const TPE_KERNEL_SHADER: &str = r#"
struct TpeUniforms {
    len: u32,
    bandwidth: f32,
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0) var<uniform> uniforms: TpeUniforms;
@group(0) @binding(1) var<storage, read> diffs_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> log_kernels_out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.len) {
        return;
    }
    let d = diffs_in[idx];
    let pi: f32 = 3.14159265;
    let log_h = log(uniforms.bandwidth);
    log_kernels_out[idx] = -0.5 * d * d - 0.5 * log(2.0 * pi) - log_h;
}
"#;

/// CPU implementation of Successive Halving top-k selection.
pub fn compute_sha_top_k_cpu(losses: &[f32], eta: usize) -> (Vec<f32>, Vec<f32>) {
    let n = losses.len();
    let k = (n / eta).max(1);

    let mut indexed: Vec<(usize, f32)> = losses.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut out_idx = vec![0.0f32; k];
    let mut out_loss = vec![0.0f32; k];
    for i in 0..k {
        out_idx[i] = indexed[i].0 as f32;
        out_loss[i] = indexed[i].1;
    }

    (out_idx, out_loss)
}

/// CPU implementation of Hyperband bracket schedule.
pub fn compute_hyperband_schedule_cpu(r_max: usize, eta: usize) -> Vec<f32> {
    let s_max = ((r_max as f64).ln() / (eta as f64).ln()).floor() as usize;
    let b = ((s_max + 1) * r_max) as f64;
    let num_brackets = s_max + 1;
    let mut schedule = vec![0.0f32; num_brackets * 2];

    for (row, s) in (0..=s_max).rev().enumerate() {
        let eta_s = (eta as f64).powi(s as i32);
        let n_s = ((b / (r_max as f64)) * (eta_s / ((s + 1) as f64))).ceil() as f32;
        let r_s = ((r_max as f64) * (eta as f64).powi(-(s as i32))) as f32;
        schedule[row * 2] = n_s;
        schedule[row * 2 + 1] = r_s;
    }

    schedule
}

/// CPU implementation of TPE log-likelihood ratio.
pub fn compute_tpe_log_ratio_cpu(
    x_query: &[f32],
    good_samples: &[f32],
    bad_samples: &[f32],
    bandwidth: f32,
) -> Vec<f32> {
    let pi = std::f64::consts::PI;
    let bw = bandwidth as f64;

    let log_kde = |q: &[f32], s: &[f32]| -> Vec<f64> {
        let n = s.len() as f64;
        let mut out = vec![0.0f64; q.len()];
        for i in 0..q.len() {
            let x = q[i] as f64;
            let mut log_terms = vec![0.0f64; s.len()];
            let mut max_term = -1e30f64;
            for j in 0..s.len() {
                let diff = (x - (s[j] as f64)) / bw;
                let term = -0.5 * diff * diff - 0.5 * (2.0 * pi).ln() - bw.ln();
                log_terms[j] = term;
                if term > max_term {
                    max_term = term;
                }
            }
            let mut sum_exp = 0.0f64;
            for j in 0..s.len() {
                sum_exp += (log_terms[j] - max_term).exp();
            }
            out[i] = max_term + sum_exp.ln() - n.ln();
        }
        out
    };

    let log_l = log_kde(x_query, good_samples);
    let log_g = log_kde(x_query, bad_samples);

    let mut ratios = vec![0.0f32; x_query.len()];
    for i in 0..x_query.len() {
        ratios[i] = (log_l[i] - log_g[i]) as f32;
    }

    ratios
}

/// CPU implementation of ASHA promotion check.
pub fn check_asha_promotion_cpu(rung_losses: &[f32], candidate_loss: f32, eta: usize) -> f32 {
    let mut losses = rung_losses.to_vec();
    let k = (losses.len() / eta).max(1);
    losses.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let cutoff = losses[k - 1];
    if candidate_loss <= cutoff {
        1.0
    } else {
        0.0
    }
}

/// Linear anchor forward pass: x @ w + b.
pub fn anchor_forward_cpu(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * out_dim];
    for r in 0..rows {
        for c in 0..out_dim {
            let mut sum = b[c];
            for k in 0..in_dim {
                sum += x[r * in_dim + k] * w[k * out_dim + c];
            }
            out[r * out_dim + c] = sum;
        }
    }
    out
}

/// Auto-dispatchers with CPU fallback.
pub fn compute_sha_top_k_auto(
    _device: &KernelDevice,
    losses: &[f32],
    eta: usize,
) -> (Vec<f32>, Vec<f32>) {
    compute_sha_top_k_cpu(losses, eta)
}

pub fn compute_hyperband_schedule_auto(
    _device: &KernelDevice,
    r_max: usize,
    eta: usize,
) -> Vec<f32> {
    compute_hyperband_schedule_cpu(r_max, eta)
}

pub fn compute_tpe_log_ratio_auto(
    _device: &KernelDevice,
    x_query: &[f32],
    good_samples: &[f32],
    bad_samples: &[f32],
    bandwidth: f32,
) -> Vec<f32> {
    compute_tpe_log_ratio_cpu(x_query, good_samples, bad_samples, bandwidth)
}

pub fn check_asha_promotion_auto(
    _device: &KernelDevice,
    rung_losses: &[f32],
    candidate_loss: f32,
    eta: usize,
) -> f32 {
    check_asha_promotion_cpu(rung_losses, candidate_loss, eta)
}
