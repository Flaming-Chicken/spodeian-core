//! CPU and WGSL kernels for Domain Generalization IRM GroupDRO VREx CORAL and DANN.

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

/// WGSL shader for parallel Deep CORAL covariance matrix difference reduction:
/// diff[r, c] = (cov_s[r, c] - cov_t[r, c])^2 / (4.0 * d^2)
pub const CORAL_DIFF_WGSL: &str = r#"
struct Params {
    dim: u32,
    norm_factor: f32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> cov_s: array<f32>;
@group(0) @binding(2) var<storage, read> cov_t: array<f32>;
@group(0) @binding(3) var<storage, read_write> diff_sq: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    let c = gid.y;
    let d = params.dim;
    if (r >= d || c >= d) {
        return;
    }
    let idx = r * d + c;
    let diff = cov_s[idx] - cov_t[idx];
    diff_sq[idx] = (diff * diff) * params.norm_factor;
}
"#;

/// CPU reference calculation for Deep CORAL loss.
pub fn deep_coral_cpu(cov_s: &[f32], cov_t: &[f32], d: usize) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..(d * d) {
        let diff = cov_s[i] - cov_t[i];
        sum += diff * diff;
    }
    sum / (4.0 * (d * d) as f32)
}

/// Automatic dispatcher falling back to CPU.
pub fn deep_coral_auto(cov_s: &[f32], cov_t: &[f32], d: usize) -> f32 {
    deep_coral_cpu(cov_s, cov_t, d)
}
