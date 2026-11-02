//! CPU (and optional WGSL) kernels for AI Analysis Loss Landscape Hessian Lanczos and Empirical NTK.

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

/// WGSL compute shader for Empirical NTK Gram matrix: Theta = J * J^T.
pub const EMPIRICAL_NTK_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> jacobians: array<f32>;
@group(0) @binding(1) var<storage, read_write> ntk_gram: array<f32>;

struct Params {
    n: u32,
    p: u32,
    _pad0: u32,
    _pad1: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let n = params.n;
    let p = params.p;
    if (idx >= n * n) {
        return;
    }
    let i = idx / n;
    let j = idx % n;

    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < p; k = k + 1u) {
        sum = sum + jacobians[i * p + k] * jacobians[j * p + k];
    }
    ntk_gram[idx] = sum;
}
"#;

/// CPU reference implementation of Empirical NTK Gram matrix.
pub fn empirical_ntk_cpu(jacobians: &[f32], n: usize, p: usize) -> Vec<f32> {
    let mut gram = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0f32;
            for k in 0..p {
                sum += jacobians[i * p + k] * jacobians[j * p + k];
            }
            gram[i * n + j] = sum;
        }
    }
    gram
}
