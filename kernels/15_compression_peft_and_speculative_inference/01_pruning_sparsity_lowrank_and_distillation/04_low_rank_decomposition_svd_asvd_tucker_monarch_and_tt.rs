//! CPU and WGSL kernels for Low Rank Decomposition SVD Asvd Tucker Monarch and Tt.

/// WGSL compute shader for two-stage low-rank factored GEMM: Y = (X @ A) @ B.
pub const LOW_RANK_GEMM_WGSL: &str = r#"
struct Uniforms {
    batch: u32,
    d_in: u32,
    rank: u32,
    d_out: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> x: array<f32>; // [batch, d_in]
@group(0) @binding(2) var<storage, read> a: array<f32>; // [d_in, rank]
@group(0) @binding(3) var<storage, read> b: array<f32>; // [rank, d_out]
@group(0) @binding(4) var<storage, read_write> y_out: array<f32>; // [batch, d_out]

@compute @workgroup_size(16, 16, 1)
fn low_rank_gemm_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b_idx = gid.x;
    let o = gid.y;
    if (b_idx >= u.batch || o >= u.d_out) { return; }

    var sum: f32 = 0.0;
    for (var r = 0u; r < u.rank; r++) {
        // Intermediate projection: h[r] = sum_i x[b, i] * a[i, r]
        var h_r: f32 = 0.0;
        for (var i = 0u; i < u.d_in; i++) {
            h_r += x[b_idx * u.d_in + i] * a[i * u.rank + r];
        }
        sum += h_r * b[r * u.d_out + o];
    }
    y_out[b_idx * u.d_out + o] = sum;
}
"#;

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

/// CPU low-rank GEMM: Y = (X @ A) @ B.
pub fn low_rank_gemm_cpu(
    x: &[f32],
    a: &[f32],
    b: &[f32],
    batch: usize,
    d_in: usize,
    rank: usize,
    d_out: usize,
) -> Vec<f32> {
    let mut h = vec![0.0f32; batch * rank];
    for b_idx in 0..batch {
        for r in 0..rank {
            let mut sum = 0.0f32;
            for i in 0..d_in {
                sum += x[b_idx * d_in + i] * a[i * rank + r];
            }
            h[b_idx * rank + r] = sum;
        }
    }

    let mut y = vec![0.0f32; batch * d_out];
    for b_idx in 0..batch {
        for o in 0..d_out {
            let mut sum = 0.0f32;
            for r in 0..rank {
                sum += h[b_idx * rank + r] * b[r * d_out + o];
            }
            y[b_idx * d_out + o] = sum;
        }
    }
    y
}

/// CPU Monarch forward transformation using two block-diagonal stages.
pub fn monarch_forward_cpu(x: &[f32], b1: &[f32], b2: &[f32], batch: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; batch * 16];
    for b in 0..batch {
        let mut h1 = [[0.0f32; 4]; 4];
        for k in 0..4 {
            for j in 0..4 {
                let mut sum = 0.0f32;
                for i in 0..4 {
                    sum += x[b * 16 + k * 4 + i] * b1[k * 16 + i * 4 + j];
                }
                h1[k][j] = sum;
            }
        }
        for k in 0..4 {
            for j in 0..4 {
                let mut sum = 0.0f32;
                for i in 0..4 {
                    sum += h1[i][k] * b2[k * 16 + i * 4 + j];
                }
                y[b * 16 + k * 4 + j] = sum;
            }
        }
    }
    y
}
