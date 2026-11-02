//! CPU (and optional WGSL) kernels for Multimodal Fusion FiLM Perceiver Resampler and VLM Connectors.

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

/// CPU kernel for FiLM modulation: y[b, l, d] = gamma[b, d] * x[b, l, d] + beta[b, d].
pub fn film_modulation_cpu(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    batch: usize,
    seq_len: usize,
    d_model: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * seq_len * d_model];
    for b in 0..batch {
        let g_slice = &gamma[b * d_model..(b + 1) * d_model];
        let b_slice = &beta[b * d_model..(b + 1) * d_model];
        for l in 0..seq_len {
            let offset = (b * seq_len + l) * d_model;
            for d in 0..d_model {
                out[offset + d] = g_slice[d] * x[offset + d] + b_slice[d];
            }
        }
    }
    out
}

/// WGSL shader for fused FiLM feature-wise linear modulation.
pub const FILM_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

struct Uniforms {
    batch: u32,
    seq_len: u32,
    d_model: u32,
};
@group(0) @binding(4) var<uniform> u: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = u.batch * u.seq_len * u.d_model;
    if (idx >= total) {
        return;
    }
    let d = idx % u.d_model;
    let b = idx / (u.seq_len * u.d_model);
    let g_val = gamma[b * u.d_model + d];
    let b_val = beta[b * u.d_model + d];
    y[idx] = g_val * x[idx] + b_val;
}
"#;
