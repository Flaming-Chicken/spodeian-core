//! CPU and WGSL kernels for Vector Product and Additive Quantization VQ PQ RVQ AQLM.

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

/// WGSL compute shader for fused AQLM multi-codebook dequantization:
/// `W_recon[n, d] = sum_{m=0}^{M-1} Codebook[m, Indices[n, m], d]`
pub const AQLM_DEQUANT_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> indices: array<u32>;
@group(0) @binding(1) var<storage, read> codebooks: array<f32>;
@group(0) @binding(2) var<storage, read_write> w_recon: array<f32>;

struct Uniforms {
    n_vecs: u32,
    n_codebooks: u32,
    codebook_size: u32,
    vec_dim: u32,
};
@group(0) @binding(3) var<uniform> params: Uniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = global_id.x;
    let d = global_id.y;
    if (n >= params.n_vecs || d >= params.vec_dim) {
        return;
    }
    var acc: f32 = 0.0;
    for (var m: u32 = 0u; m < params.n_codebooks; m = m + 1u) {
        let idx = indices[n * params.n_codebooks + m];
        let cb_offset = (m * params.codebook_size + idx) * params.vec_dim + d;
        acc = acc + codebooks[cb_offset];
    }
    w_recon[n * params.vec_dim + d] = acc;
}
"#;

/// CPU reference implementation of fused AQLM dequantization.
pub fn aqlm_dequant_cpu(
    indices: &[u32],
    codebooks: &[f32],
    n_vecs: usize,
    n_codebooks: usize,
    codebook_size: usize,
    vec_dim: usize,
) -> Vec<f32> {
    let mut w_recon = vec![0.0f32; n_vecs * vec_dim];
    for n in 0..n_vecs {
        for d in 0..vec_dim {
            let mut acc = 0.0f32;
            for m in 0..n_codebooks {
                let idx = indices[n * n_codebooks + m] as usize;
                let cb_offset = (m * codebook_size + idx) * vec_dim + d;
                acc += codebooks[cb_offset];
            }
            w_recon[n * vec_dim + d] = acc;
        }
    }
    w_recon
}

/// Auto dispatcher: returns CPU reference fallback.
pub fn aqlm_dequant_auto(
    indices: &[u32],
    codebooks: &[f32],
    n_vecs: usize,
    n_codebooks: usize,
    codebook_size: usize,
    vec_dim: usize,
) -> Vec<f32> {
    aqlm_dequant_cpu(
        indices,
        codebooks,
        n_vecs,
        n_codebooks,
        codebook_size,
        vec_dim,
    )
}
