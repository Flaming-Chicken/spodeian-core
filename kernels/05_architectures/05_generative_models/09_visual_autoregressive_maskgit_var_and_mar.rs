//! CPU (and optional WGSL) kernels for Visual Autoregressive MaskGIT, VAR, and MAR.

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

/// Applies binary mask M in {0, 1} to replace masked token features with mask_token vector.
pub fn maskgit_apply_mask_cpu(
    x: &[f32],
    mask: &[f32],
    mask_token: &[f32],
    batch: usize,
    seq_len: usize,
    dim: usize,
) -> Vec<f32> {
    let total = batch * seq_len * dim;
    let mut out = vec![0.0f32; total];

    for b in 0..batch {
        for l in 0..seq_len {
            let m_val = mask[b * seq_len + l];
            let one_minus_m = 1.0 - m_val;
            let offset = (b * seq_len + l) * dim;
            for d in 0..dim {
                out[offset + d] = one_minus_m * x[offset + d] + m_val * mask_token[d];
            }
        }
    }
    out
}

/// Computes MAR masked diffusion loss on CPU.
pub fn mar_masked_diffusion_loss_cpu(
    v_pred: &[f32],
    v_target: &[f32],
    mask: &[f32],
    batch: usize,
    seq_len: usize,
    dim: usize,
) -> f32 {
    let mut sum_sq_err = 0.0f32;
    let mut mask_sum = 0.0f32;

    for b in 0..batch {
        for l in 0..seq_len {
            let m = mask[b * seq_len + l];
            mask_sum += m;
            let offset = (b * seq_len + l) * dim;
            for d in 0..dim {
                let diff = v_pred[offset + d] - v_target[offset + d];
                sum_sq_err += m * (diff * diff);
            }
        }
    }
    sum_sq_err / (mask_sum * (dim as f32) + 1e-6)
}

/// WGSL shader for elementwise masked token injection.
pub const WGSL_MASKGIT_APPLY_MASK: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> mask: array<f32>;
@group(0) @binding(2) var<storage, read> mask_token: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

struct Uniforms {
    batch_size: u32,
    seq_len: u32,
    dim: u32,
};
@group(0) @binding(4) var<uniform> u: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = u.batch_size * u.seq_len * u.dim;
    if (idx >= total) {
        return;
    }
    let d = idx % u.dim;
    let bl = idx / u.dim;
    let m = mask[bl];

    out[idx] = (1.0 - m) * x[idx] + m * mask_token[d];
}
"#;
