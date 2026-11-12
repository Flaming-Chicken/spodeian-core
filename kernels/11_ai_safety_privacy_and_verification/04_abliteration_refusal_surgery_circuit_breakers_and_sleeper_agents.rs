//! CPU (and optional WGSL) kernels for Abliteration Refusal Surgery Circuit Breakers and Sleeper Agents.

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

/// WGSL compute shader for Contrastive Activation Addition (CAA / RepE dynamic steering).
pub const CONTRASTIVE_ACTIVATION_ADDITION_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> h_in: array<f32>;
@group(0) @binding(1) var<storage, read> steer_vec: array<f32>;
@group(0) @binding(2) var<storage, read_write> h_out: array<f32>;

struct Params {
    batch: u32,
    d_model: u32,
    alpha: f32,
    _pad0: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = params.batch * params.d_model;
    let idx = gid.x;
    if (idx >= total) {
        return;
    }

    let d_idx = idx % params.d_model;
    h_out[idx] = h_in[idx] + params.alpha * steer_vec[d_idx];
}
"#;

/// CPU reference implementation of Contrastive Activation Addition.
pub fn contrastive_activation_addition_cpu(
    h: &[f32],
    steer_vec: &[f32],
    alpha: f32,
    batch: usize,
    d_model: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * d_model];
    for b in 0..batch {
        for d in 0..d_model {
            let idx = b * d_model + d;
            out[idx] = h[idx] + alpha * steer_vec[d];
        }
    }
    out
}
