//! CPU (and optional WGSL) kernels for Topological Data Analysis Persistent Homology and Singular Learning LLC.

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

/// WGSL compute shader for evaluating Level 1 persistence landscape across grid points.
pub const PERSISTENCE_LANDSCAPE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> pairs: array<f32>;
@group(0) @binding(1) var<storage, read> eval_points: array<f32>;
@group(0) @binding(2) var<storage, read_write> landscape: array<f32>;

struct Params {
    m: u32,
    t_count: u32,
    _pad0: u32,
    _pad1: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let t_count = params.t_count;
    let m = params.m;
    if (col >= t_count) {
        return;
    }

    let t = eval_points[col];
    var max_val: f32 = 0.0;
    for (var i: u32 = 0u; i < m; i = i + 1u) {
        let b = pairs[i * 2u + 0u];
        let d = pairs[i * 2u + 1u];
        let term1 = t - b;
        let term2 = d - t;
        let val = max(0.0, min(term1, term2));
        if (val > max_val) {
            max_val = val;
        }
    }
    landscape[col] = max_val;
}
"#;

/// CPU reference implementation of Level 1 persistence landscape evaluation.
pub fn persistence_landscape_cpu(pairs: &[f32], eval_points: &[f32], m: usize) -> Vec<f32> {
    let mut landscape = vec![0.0f32; eval_points.len()];
    for (col, &t) in eval_points.iter().enumerate() {
        let mut max_val = 0.0f32;
        for i in 0..m {
            let b = pairs[i * 2];
            let d = pairs[i * 2 + 1];
            let val = (t - b).min(d - t).max(0.0);
            if val > max_val {
                max_val = val;
            }
        }
        landscape[col] = max_val;
    }
    landscape
}
