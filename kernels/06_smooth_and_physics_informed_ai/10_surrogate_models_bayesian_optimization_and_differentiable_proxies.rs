//! CPU (and optional WGSL) kernels for Surrogate Models Bayesian Optimization & Differentiable Proxies.

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

/// Fused RBF kernel computation on CPU:
/// K[i, j] = variance * exp( -0.5 * sum_d ((x1[i, d] - x2[j, d]) / lengthscale[d])^2 )
pub fn rbf_kernel_cpu(
    x1: &[f32],
    x2: &[f32],
    lengthscale: &[f32],
    variance: f32,
    n1: usize,
    n2: usize,
    d: usize,
) -> Vec<f32> {
    let mut k = vec![0.0f32; n1 * n2];
    for i in 0..n1 {
        for j in 0..n2 {
            let mut dist_sq = 0.0f32;
            for col in 0..d {
                let diff = (x1[i * d + col] - x2[j * d + col]) / lengthscale[col];
                dist_sq += diff * diff;
            }
            k[i * n2 + j] = variance * (-0.5f32 * dist_sq).exp();
        }
    }
    k
}

/// WGSL compute shader for parallel pairwise RBF kernel evaluation.
pub const RBF_KERNEL_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> x1_buf: array<f32>;
@group(0) @binding(1) var<storage, read> x2_buf: array<f32>;
@group(0) @binding(2) var<storage, read> lengthscale_buf: array<f32>;
@group(0) @binding(3) var<storage, read_write> k_out_buf: array<f32>;

struct KernelDims {
    n1: u32,
    n2: u32,
    d: u32,
    variance: f32,
};
@group(0) @binding(4) var<uniform> dims: KernelDims;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let j = global_id.y;

    if (i >= dims.n1 || j >= dims.n2) {
        return;
    }

    var dist_sq: f32 = 0.0;
    for (var col: u32 = 0u; col < dims.d; col = col + 1u) {
        let diff = (x1_buf[i * dims.d + col] - x2_buf[j * dims.d + col]) / lengthscale_buf[col];
        dist_sq = dist_sq + diff * diff;
    }

    let idx = i * dims.n2 + j;
    k_out_buf[idx] = dims.variance * exp(-0.5 * dist_sq);
}
"#;
