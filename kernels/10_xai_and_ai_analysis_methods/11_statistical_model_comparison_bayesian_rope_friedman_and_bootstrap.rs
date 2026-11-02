//! CPU (and optional WGSL) kernels for Statistical Model Comparison Bayesian RoPE Friedman and Bootstrap.

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

/// WGSL compute shader for column-wise average ranks reduction across N benchmarks for K algorithms.
pub const AVERAGE_RANKS_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> ranks: array<f32>;
@group(0) @binding(1) var<storage, read_write> avg_ranks: array<f32>;

struct Params {
    n: u32,
    k: u32,
    _pad0: u32,
    _pad1: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let n = params.n;
    let k = params.k;
    if (col >= k) {
        return;
    }

    var sum: f32 = 0.0;
    for (var r: u32 = 0u; r < n; r = r + 1u) {
        sum = sum + ranks[r * k + col];
    }
    avg_ranks[col] = sum / f32(n);
}
"#;

/// CPU reference implementation of column-wise average ranks reduction.
pub fn average_ranks_cpu(ranks: &[f32], n: usize, k: usize) -> Vec<f32> {
    let mut avg = vec![0.0f32; k];
    for r in 0..n {
        for c in 0..k {
            avg[c] += ranks[r * k + c];
        }
    }
    for c in 0..k {
        avg[c] /= n as f32;
    }
    avg
}
