//! CPU (and optional WGSL) kernels for Quality Diversity Map Elites and Differentiable Qd.

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

/// WGSL compute shader for DQD gradient arborescence step across parameter dimensions.
pub const DQD_ARBORESCENCE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> grad_f: array<f32>;
@group(0) @binding(1) var<storage, read> jac_b: array<f32>;
@group(0) @binding(2) var<storage, read> coeffs: array<f32>;
@group(0) @binding(3) var<storage, read_write> step_out: array<f32>;

struct Params {
    d_dim: u32,
    k_desc: u32,
    _pad0: u32,
    _pad1: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = gid.x;
    if (d >= params.d_dim) {
        return;
    }
    var val: f32 = coeffs[0] * grad_f[d];
    for (var k: u32 = 0u; k < params.k_desc; k = k + 1u) {
        let c = coeffs[k + 1u];
        val = val + c * jac_b[k * params.d_dim + d];
    }
    step_out[d] = val;
}
"#;

/// CPU reference implementation of DQD gradient arborescence step.
pub fn dqd_arborescence_cpu(
    grad_f: &[f32],
    jac_b: &[f32],
    coeffs: &[f32],
    d_dim: usize,
    k_desc: usize,
) -> Vec<f32> {
    let mut step = vec![0.0f32; d_dim];
    for d in 0..d_dim {
        let mut val = coeffs[0] * grad_f[d];
        for k in 0..k_desc {
            val += coeffs[k + 1] * jac_b[k * d_dim + d];
        }
        step[d] = val;
    }
    step
}

/// Auto-dispatching DQD gradient arborescence step (CPU fallback).
pub fn dqd_arborescence_auto(
    grad_f: &[f32],
    jac_b: &[f32],
    coeffs: &[f32],
    d_dim: usize,
    k_desc: usize,
) -> Vec<f32> {
    dqd_arborescence_cpu(grad_f, jac_b, coeffs, d_dim, k_desc)
}
