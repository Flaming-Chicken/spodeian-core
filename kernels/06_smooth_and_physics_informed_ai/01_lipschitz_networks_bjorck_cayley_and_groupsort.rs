//! CPU (and optional WGSL) kernels for Lipschitz Networks Bjorck Cayley and Groupsort.

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

/// CPU kernel for GroupSort-2 (PairSort) activation function:
/// Sorts adjacent pairs (x[2i], x[2i+1]) into (max, min).
pub fn groupsort2_cpu(x: &[f32], batch: usize, dim: usize) -> Vec<f32> {
    assert_eq!(dim % 2, 0, "Dimension must be even for GroupSort-2");
    let mut y = vec![0.0f32; batch * dim];
    for b in 0..batch {
        for p in 0..(dim / 2) {
            let idx0 = b * dim + 2 * p;
            let idx1 = b * dim + 2 * p + 1;
            let a = x[idx0];
            let b_val = x[idx1];
            if a >= b_val {
                y[idx0] = a;
                y[idx1] = b_val;
            } else {
                y[idx0] = b_val;
                y[idx1] = a;
            }
        }
    }
    y
}

/// WGSL shader for fused GroupSort-2 (PairSort) activation.
pub const GROUPSORT2_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;

struct Uniforms {
    total_pairs: u32,
};
@group(0) @binding(2) var<uniform> u: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pair_idx = global_id.x;
    if (pair_idx >= u.total_pairs) {
        return;
    }
    let idx0 = pair_idx * 2u;
    let idx1 = idx0 + 1u;
    let a = x[idx0];
    let b = x[idx1];
    if (a >= b) {
        y[idx0] = a;
        y[idx1] = b;
    } else {
        y[idx0] = b;
        y[idx1] = a;
    }
}
"#;
