//! CPU (and optional WGSL) kernels for AI Analysis Representation Similarity CKA SVCCA RSA and Intrinsic Dim.

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

/// WGSL compute shader for Gram matrix double centering.
pub const CENTER_GRAM_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> gram: array<f32>;
@group(0) @binding(1) var<storage, read> row_means: array<f32>;
@group(0) @binding(2) var<storage, read> col_means: array<f32>;
@group(0) @binding(3) var<storage, read_write> centered: array<f32>;

struct Params {
    n: u32,
    total_mean: f32,
    _pad0: u32,
    _pad1: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let n = params.n;
    if (idx >= n * n) {
        return;
    }
    let r = idx / n;
    let c = idx % n;
    centered[idx] = gram[idx] - row_means[c] - col_means[r] + params.total_mean;
}
"#;

/// CPU reference implementation of Gram matrix double centering.
pub fn center_gram_cpu(
    gram: &[f32],
    row_means: &[f32],
    col_means: &[f32],
    total_mean: f32,
    n: usize,
) -> Vec<f32> {
    let mut centered = vec![0.0f32; n * n];
    for r in 0..n {
        for c in 0..n {
            let idx = r * n + c;
            centered[idx] = gram[idx] - row_means[c] - col_means[r] + total_mean;
        }
    }
    centered
}
