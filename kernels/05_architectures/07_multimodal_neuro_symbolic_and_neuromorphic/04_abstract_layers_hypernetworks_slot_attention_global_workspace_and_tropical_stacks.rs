//! CPU (and optional WGSL) kernels for Abstract Layers Hypernetworks Slot Attention Global Workspace and Tropical Stacks.

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

/// CPU kernel for tropical (max-plus) semiring matrix multiplication:
/// Y[b, j] = max_k (X[b, k] + W[k, j]).
pub fn tropical_max_plus_matmul_cpu(
    x: &[f32],
    w: &[f32],
    batch: usize,
    d_in: usize,
    d_out: usize,
) -> Vec<f32> {
    let mut y = vec![f32::NEG_INFINITY; batch * d_out];
    for b in 0..batch {
        for j in 0..d_out {
            let mut max_val = f32::NEG_INFINITY;
            for k in 0..d_in {
                let val = x[b * d_in + k] + w[k * d_out + j];
                if val > max_val {
                    max_val = val;
                }
            }
            y[b * d_out + j] = max_val;
        }
    }
    y
}

/// WGSL compute shader for tropical (max-plus) semiring matrix multiplication.
pub const TROPICAL_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

struct Uniforms {
    batch: u32,
    d_in: u32,
    d_out: u32,
};
@group(0) @binding(3) var<uniform> u: Uniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.x;
    let j = global_id.y;
    if (b >= u.batch || j >= u.d_out) {
        return;
    }
    var max_val: f32 = -1e30;
    for (var k: u32 = 0u; k < u.d_in; k = k + 1u) {
        let val = x[b * u.d_in + k] + w[k * u.d_out + j];
        if (val > max_val) {
            max_val = val;
        }
    }
    y[b * u.d_out + j] = max_val;
}
"#;
