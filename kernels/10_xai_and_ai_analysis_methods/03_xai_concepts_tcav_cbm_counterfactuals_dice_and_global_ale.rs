//! CPU (and optional WGSL) kernels for XAI Concepts TCAV CBM Counterfactuals DiCE and Global ALE.

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

/// WGSL compute shader for Concept Bottleneck Model test-time concept intervention.
pub const CBM_INTERVENTION_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> pred_concepts: array<f32>;
@group(0) @binding(1) var<storage, read> true_concepts: array<f32>;
@group(0) @binding(2) var<storage, read> intervention_mask: array<f32>;
@group(0) @binding(3) var<storage, read_write> intervened_concepts: array<f32>;

struct Params {
    num_samples: u32,
    num_concepts: u32,
    _pad0: u32,
    _pad1: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total_elements = params.num_samples * params.num_concepts;
    if (idx >= total_elements) {
        return;
    }
    let k = idx % params.num_concepts;
    let mask = intervention_mask[k];
    let pred_c = pred_concepts[idx];
    let true_c = true_concepts[idx];
    intervened_concepts[idx] = (1.0 - mask) * pred_c + mask * true_c;
}
"#;

/// CPU reference implementation of Concept Bottleneck Model test-time concept intervention.
pub fn cbm_intervention_cpu(
    pred_concepts: &[f32],
    true_concepts: &[f32],
    intervention_mask: &[f32],
    num_samples: usize,
    num_concepts: usize,
) -> Vec<f32> {
    let total = num_samples * num_concepts;
    let mut out = vec![0.0f32; total];
    for n in 0..num_samples {
        for k in 0..num_concepts {
            let idx = n * num_concepts + k;
            let m = intervention_mask[k];
            out[idx] = (1.0 - m) * pred_concepts[idx] + m * true_concepts[idx];
        }
    }
    out
}
