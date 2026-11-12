//! CPU (and optional WGSL) kernels for Discrete Diffusion D3PM, MDLM, and SEDD.

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

/// Continuous-time Masked Diffusion Language Model (MDLM) marginal forward pass on CPU.
pub fn mdlm_marginal_cpu(
    x_0: &[f32],
    alpha: &[f32],
    batch: usize,
    seq_len: usize,
    vocab_size: usize,
    mask_idx: usize,
) -> Vec<f32> {
    let total = batch * seq_len * vocab_size;
    let mut x_t = vec![0.0f32; total];

    for b in 0..batch {
        let a = alpha[b];
        let one_minus_a = 1.0 - a;
        for l in 0..seq_len {
            let row_offset = (b * seq_len + l) * vocab_size;
            for k in 0..vocab_size {
                let mask_val = if k == mask_idx { 1.0 } else { 0.0 };
                x_t[row_offset + k] = a * x_0[row_offset + k] + one_minus_a * mask_val;
            }
        }
    }
    x_t
}

/// WGSL shader for continuous-time masked diffusion forward marginal token interpolation.
pub const WGSL_MDLM_MARGINAL: &str = r#"
@group(0) @binding(0) var<storage, read> x_0: array<f32>;
@group(0) @binding(1) var<storage, read> alpha_t: array<f32>;
@group(0) @binding(2) var<storage, read_write> x_t: array<f32>;

struct Uniforms {
    batch_size: u32,
    seq_len: u32,
    vocab_size: u32,
    mask_idx: u32,
};
@group(0) @binding(3) var<uniform> u: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = u.batch_size * u.seq_len * u.vocab_size;
    if (idx >= total) {
        return;
    }
    let k = idx % u.vocab_size;
    let b = idx / (u.seq_len * u.vocab_size);

    let a = alpha_t[b];
    let mask_val = select(0.0, 1.0, k == u.mask_idx);

    x_t[idx] = a * x_0[idx] + (1.0 - a) * mask_val;
}
"#;
