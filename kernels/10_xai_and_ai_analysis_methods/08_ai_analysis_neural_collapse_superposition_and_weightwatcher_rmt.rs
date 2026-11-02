//! CPU (and optional WGSL) kernels for AI Analysis Neural Collapse Superposition and WeightWatcher RMT.

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

/// WGSL compute shader for superposition feature interference Gram matrix: G = W^T W.
pub const SUPERPOSITION_GRAM_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> w: array<f32>;
@group(0) @binding(1) var<storage, read_write> gram: array<f32>;

struct Params {
    d: u32,
    m: u32,
    _pad0: u32,
    _pad1: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let m = params.m;
    let d = params.d;
    if (idx >= m * m) {
        return;
    }
    let i = idx / m;
    let j = idx % m;

    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < d; k = k + 1u) {
        let w_ki = w[k * m + i];
        let w_kj = w[k * m + j];
        sum = sum + w_ki * w_kj;
    }
    gram[idx] = sum;
}
"#;

/// CPU reference implementation of superposition feature interference and Gram matrix.
pub fn superposition_gram_cpu(w: &[f32], d: usize, m: usize) -> (Vec<f32>, f32) {
    let mut gram = vec![0.0f32; m * m];
    let mut interf = 0.0f32;
    for i in 0..m {
        for j in 0..m {
            let mut sum = 0.0f32;
            for k in 0..d {
                sum += w[k * m + i] * w[k * m + j];
            }
            gram[i * m + j] = sum;
            if i != j {
                interf += sum * sum;
            }
        }
    }
    (gram, interf)
}
