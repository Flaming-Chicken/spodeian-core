//! Hardware Kernels (WGSL + Native CPU fallback) for Category 04 Topic 11:
//! 04_combinatorial_and_discrete_simulated_annealing_tempering_and_sinkhorn in 04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation.

use crate::context::KernelDevice;

/// WGSL Compute Shader for Simulated Annealing Metropolis-Hastings acceptance: min(1, exp(-delta_e / T)).
pub const SA_ACCEPTANCE_SHADER: &str = r#"
struct SaUniforms {
    len: u32,
    temp: f32,
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0) var<uniform> uniforms: SaUniforms;
@group(0) @binding(1) var<storage, read> delta_e_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> probs_out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.len) {
        return;
    }
    let de = delta_e_in[idx];
    let p = exp(-de / uniforms.temp);
    probs_out[idx] = min(1.0, p);
}
"#;

/// CPU implementation of Sinkhorn normalization on square matrix.
pub fn run_sinkhorn_cpu(
    log_alpha: &[f32],
    rows: usize,
    cols: usize,
    n_iters: usize,
    tau: f32,
) -> Vec<f32> {
    let mut p = vec![0.0f32; rows * cols];
    for i in 0..(rows * cols) {
        p[i] = (log_alpha[i] / tau).exp();
    }

    let eps = 1e-12f32;

    for _ in 0..n_iters {
        // Row normalization
        for r in 0..rows {
            let mut row_sum = eps;
            for c in 0..cols {
                row_sum += p[r * cols + c];
            }
            for c in 0..cols {
                p[r * cols + c] /= row_sum;
            }
        }

        // Column normalization
        for c in 0..cols {
            let mut col_sum = eps;
            for r in 0..rows {
                col_sum += p[r * cols + c];
            }
            for r in 0..rows {
                p[r * cols + c] /= col_sum;
            }
        }
    }

    p
}

/// CPU implementation of Cross-Entropy Method (CEM) step.
pub fn step_cem_cpu(
    candidates: &[f32],
    scores: &[f32],
    n_samples: usize,
    dim: usize,
    elite_frac: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let n_elite = ((n_samples as f32) * elite_frac).floor() as usize;
    let n_elite = n_elite.max(1);

    let mut mean = vec![0.0f32; dim];
    for &(idx, _) in indexed.iter().take(n_elite) {
        for j in 0..dim {
            mean[j] += candidates[idx * dim + j];
        }
    }
    for j in 0..dim {
        mean[j] /= n_elite as f32;
    }

    let mut std = vec![0.0f32; dim];
    for &(idx, _) in indexed.iter().take(n_elite) {
        for j in 0..dim {
            let diff = candidates[idx * dim + j] - mean[j];
            std[j] += diff * diff;
        }
    }
    for j in 0..dim {
        std[j] = (std[j] / (n_elite as f32)).sqrt() + 1e-6;
    }

    (mean, std)
}

/// CPU implementation of Simulated Annealing acceptance probability.
pub fn sa_acceptance_prob_cpu(delta_e: &[f32], temp: f32) -> Vec<f32> {
    delta_e
        .iter()
        .map(|&de| (-(de as f64) / (temp as f64)).exp().min(1.0) as f32)
        .collect()
}

/// CPU implementation of Parallel Tempering replica exchange probability.
pub fn parallel_tempering_swap_prob_cpu(e1: f32, e2: f32, temp1: f32, temp2: f32) -> f32 {
    let beta1 = 1.0f64 / (temp1 as f64);
    let beta2 = 1.0f64 / (temp2 as f64);
    let delta = (beta1 - beta2) * ((e1 as f64) - (e2 as f64));
    delta.exp().min(1.0) as f32
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
pub fn run_sinkhorn_auto(
    _device: &KernelDevice,
    log_alpha: &[f32],
    rows: usize,
    cols: usize,
    n_iters: usize,
    tau: f32,
) -> Vec<f32> {
    run_sinkhorn_cpu(log_alpha, rows, cols, n_iters, tau)
}

pub fn step_cem_auto(
    _device: &KernelDevice,
    candidates: &[f32],
    scores: &[f32],
    n_samples: usize,
    dim: usize,
    elite_frac: f32,
) -> (Vec<f32>, Vec<f32>) {
    step_cem_cpu(candidates, scores, n_samples, dim, elite_frac)
}

pub fn sa_acceptance_prob_auto(_device: &KernelDevice, delta_e: &[f32], temp: f32) -> Vec<f32> {
    sa_acceptance_prob_cpu(delta_e, temp)
}

pub fn parallel_tempering_swap_prob_auto(
    _device: &KernelDevice,
    e1: f32,
    e2: f32,
    temp1: f32,
    temp2: f32,
) -> f32 {
    parallel_tempering_swap_prob_cpu(e1, e2, temp1, temp2)
}
