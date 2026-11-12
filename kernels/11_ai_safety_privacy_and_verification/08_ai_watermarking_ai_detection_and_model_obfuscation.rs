//! CPU (and optional WGSL) kernels for AI Watermarking AI Detection and Model Obfuscation.

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

/// WGSL compute shader for Kirchenbauer green-list logit biasing:
/// l' = l + delta * green_mask.
pub const WATERMARK_LOGIT_BIAS_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> logits_in: array<f32>;
@group(0) @binding(1) var<storage, read> green_mask: array<f32>;
@group(0) @binding(2) var<storage, read_write> logits_out: array<f32>;

struct Params {
    batch: u32,
    vocab_size: u32,
    delta: f32,
    _pad: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = params.batch * params.vocab_size;
    let idx = gid.x;
    if (idx >= total) {
        return;
    }

    let v_idx = idx % params.vocab_size;
    logits_out[idx] = logits_in[idx] + params.delta * green_mask[v_idx];
}
"#;

/// CPU reference implementation of green-list logit biasing.
pub fn watermark_logit_bias_cpu(
    logits: &[f32],
    green_mask: &[f32],
    delta: f32,
    batch: usize,
    vocab_size: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * vocab_size];
    for b in 0..batch {
        for v in 0..vocab_size {
            let idx = b * vocab_size + v;
            out[idx] = logits[idx] + delta * green_mask[v];
        }
    }
    out
}
