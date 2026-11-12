//! CPU (and optional WGSL) kernels for XAI Local Surrogates LIME SHAP and Perturbation Masks.

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

/// WGSL compute shader for RISE feature importance reduction across perturbation samples.
pub const RISE_IMPORTANCE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> masks: array<f32>;
@group(0) @binding(1) var<storage, read> model_outputs: array<f32>;
@group(0) @binding(2) var<storage, read_write> importance: array<f32>;

struct Params {
    num_samples: u32,
    num_features: u32,
    _pad0: u32,
    _pad1: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m = gid.x;
    if (m >= params.num_features) {
        return;
    }
    var sum_val: f32 = 0.0;
    for (var k: u32 = 0u; k < params.num_samples; k = k + 1u) {
        let y = model_outputs[k];
        let mask_val = masks[k * params.num_features + m];
        sum_val = sum_val + y * mask_val;
    }
    importance[m] = sum_val / f32(params.num_samples);
}
"#;

/// CPU reference implementation of RISE feature importance reduction.
pub fn rise_importance_cpu(
    masks: &[f32],
    model_outputs: &[f32],
    num_samples: usize,
    num_features: usize,
) -> Vec<f32> {
    let mut importance = vec![0.0f32; num_features];
    let n = num_samples as f32;
    for m in 0..num_features {
        let mut sum_val = 0.0f32;
        for k in 0..num_samples {
            sum_val += model_outputs[k] * masks[k * num_features + m];
        }
        importance[m] = sum_val / n;
    }
    importance
}

/// Auto-dispatching RISE feature importance reduction (CPU fallback).
pub fn rise_importance_auto(
    masks: &[f32],
    model_outputs: &[f32],
    num_samples: usize,
    num_features: usize,
) -> Vec<f32> {
    rise_importance_cpu(masks, model_outputs, num_samples, num_features)
}
