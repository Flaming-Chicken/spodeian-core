//! CPU (and optional WGSL) kernels for Privacy DP SGD PATE Gradient Inversion DLG and MIA.

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

/// WGSL compute shader for DP-SGD per-sample gradient clipping:
/// g'_i = g_i * min(1, C / ||g_i||_2).
pub const DP_SGD_PER_SAMPLE_CLIP_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> g_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> g_out: array<f32>;

struct Params {
    batch: u32,
    d_feat: u32,
    clip_norm: f32,
    _pad: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if (b >= params.batch) {
        return;
    }
    let d = params.d_feat;
    var norm_sq: f32 = 0.0;
    for (var j: u32 = 0u; j < d; j = j + 1u) {
        let val = g_in[b * d + j];
        norm_sq = norm_sq + val * val;
    }
    let norm = sqrt(norm_sq) + 1e-8;
    let scale = min(1.0, params.clip_norm / norm);
    for (var j: u32 = 0u; j < d; j = j + 1u) {
        g_out[b * d + j] = g_in[b * d + j] * scale;
    }
}
"#;

/// CPU reference implementation of per-sample gradient clipping.
pub fn dp_sgd_per_sample_clip_cpu(
    grads: &[f32],
    batch: usize,
    d_feat: usize,
    clip_norm: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * d_feat];
    for b in 0..batch {
        let mut norm_sq = 0.0f32;
        for d in 0..d_feat {
            let val = grads[b * d_feat + d];
            norm_sq += val * val;
        }
        let norm = norm_sq.sqrt() + 1e-8f32;
        let scale = (clip_norm / norm).min(1.0f32);
        for d in 0..d_feat {
            out[b * d_feat + d] = grads[b * d_feat + d] * scale;
        }
    }
    out
}
