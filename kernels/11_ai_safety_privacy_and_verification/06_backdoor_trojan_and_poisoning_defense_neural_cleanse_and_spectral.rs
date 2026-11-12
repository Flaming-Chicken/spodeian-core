//! CPU (and optional WGSL) kernels for Backdoor Trojan and Poisoning Defense Neural Cleanse and Spectral.

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

/// WGSL compute shader for BadNets trojan trigger blending:
/// x' = x * (1 - mask) + pattern * mask.
pub const TROJAN_TRIGGER_APPLY_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> x_in: array<f32>;
@group(0) @binding(1) var<storage, read> mask: array<f32>;
@group(0) @binding(2) var<storage, read> pattern: array<f32>;
@group(0) @binding(3) var<storage, read_write> x_out: array<f32>;

struct Params {
    batch: u32,
    d_feat: u32,
    _pad0: u32,
    _pad1: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = params.batch * params.d_feat;
    let idx = gid.x;
    if (idx >= total) {
        return;
    }

    let d_idx = idx % params.d_feat;
    let m = mask[d_idx];
    let p = pattern[d_idx];
    x_out[idx] = x_in[idx] * (1.0 - m) + p * m;
}
"#;

/// CPU reference implementation of trojan trigger application.
pub fn trojan_trigger_apply_cpu(
    x: &[f32],
    mask: &[f32],
    pattern: &[f32],
    batch: usize,
    d_feat: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * d_feat];
    for b in 0..batch {
        for d in 0..d_feat {
            let idx = b * d_feat + d;
            out[idx] = x[idx] * (1.0 - mask[d]) + pattern[d] * mask[d];
        }
    }
    out
}
