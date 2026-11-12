//! CPU (and optional WGSL) kernels for Automatic Mixed Precision Autocast and Gradscaler.

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

/// WGSL compute shader for unscaling and applying norm-bounded clipping.
pub const UNSCALE_AND_CLIP_GRADS_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> g_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> g_out: array<f32>;

struct Params {
    total_elements: u32,
    inv_scale: f32,
    clip_coeff: f32,
    _pad0: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn unscale_clip_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.total_elements) {
        return;
    }

    let unscaled = g_in[idx] * params.inv_scale;
    g_out[idx] = unscaled * params.clip_coeff;
}
"#;

/// CPU reference implementation for unscaling and clipping.
pub fn unscale_and_clip_grads_cpu(
    scaled_grads: &[f32],
    scale: f32,
    max_norm: f32,
    eps: f32,
) -> (Vec<f32>, f32) {
    let inv_scale = 1.0f32 / scale;
    let unscaled: Vec<f32> = scaled_grads.iter().map(|&v| v * inv_scale).collect();
    let norm_sq: f32 = unscaled.iter().map(|&v| v * v).sum();
    let total_norm = norm_sq.sqrt();
    let coeff = (max_norm / (total_norm + eps)).min(1.0f32);
    let clipped: Vec<f32> = unscaled.iter().map(|&v| v * coeff).collect();
    (clipped, total_norm)
}
