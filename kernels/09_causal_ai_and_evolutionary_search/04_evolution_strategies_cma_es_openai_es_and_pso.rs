//! CPU (and optional WGSL) kernels for Evolution Strategies CMA Es Openai Es and Pso.

/// CPU linear anchor: `y[b, o] = bias[o] + sum_i x[b, i] * w[i, o]`.
pub fn linear_anchor_cpu(
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    batch: usize,
    d_in: usize,
    d_out: usize,
) -> Vec<f32> {
    let mut y = vec![0.0f32; batch * d_out];
    for (b, row) in y.chunks_mut(d_out).enumerate() {
        for (o, out) in row.iter_mut().enumerate() {
            *out = bias[o]
                + (0..d_in)
                    .map(|i| x[b * d_in + i] * w[i * d_out + o])
                    .sum::<f32>();
        }
    }
    y
}

/// WGSL shader for OpenAI ES antithetic gradient reduction across population perturbations.
pub const OPENAI_ES_ANTITHETIC_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> epsilons: array<f32>;
@group(0) @binding(1) var<storage, read> pos_fit: array<f32>;
@group(0) @binding(2) var<storage, read> neg_fit: array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_out: array<f32>;

struct Params {
    p_pop: u32,
    d_dim: u32,
    sigma: f32,
    _pad: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = gid.x;
    if (d >= params.d_dim) {
        return;
    }
    var sum_val: f32 = 0.0;
    for (var p: u32 = 0u; p < params.p_pop; p = p + 1u) {
        let diff = pos_fit[p] - neg_fit[p];
        let eps = epsilons[p * params.d_dim + d];
        sum_val = sum_val + diff * eps;
    }
    let scale = 1.0 / (2.0 * f32(params.p_pop) * params.sigma);
    grad_out[d] = sum_val * scale;
}
"#;

/// CPU reference implementation of OpenAI ES antithetic gradient estimation.
pub fn openai_es_antithetic_cpu(
    epsilons: &[f32],
    pos_fit: &[f32],
    neg_fit: &[f32],
    p_pop: usize,
    d_dim: usize,
    sigma: f32,
) -> Vec<f32> {
    let mut grad = vec![0.0f32; d_dim];
    let scale = 1.0 / (2.0 * (p_pop as f32) * sigma);
    for d in 0..d_dim {
        let mut sum_val = 0.0f32;
        for p in 0..p_pop {
            let diff = pos_fit[p] - neg_fit[p];
            let eps = epsilons[p * d_dim + d];
            sum_val += diff * eps;
        }
        grad[d] = sum_val * scale;
    }
    grad
}

/// Auto-dispatching OpenAI ES antithetic gradient estimation (CPU fallback).
pub fn openai_es_antithetic_auto(
    epsilons: &[f32],
    pos_fit: &[f32],
    neg_fit: &[f32],
    p_pop: usize,
    d_dim: usize,
    sigma: f32,
) -> Vec<f32> {
    openai_es_antithetic_cpu(epsilons, pos_fit, neg_fit, p_pop, d_dim, sigma)
}
