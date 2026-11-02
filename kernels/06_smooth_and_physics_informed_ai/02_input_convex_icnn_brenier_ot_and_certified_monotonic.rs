//! CPU (and optional WGSL) kernels for Input Convex ICNN Brenier OT and Certified Monotonic.

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

/// Fused CPU kernel for ICNN intermediate activation: z = relu(x * W_x + b).
pub fn icnn_layer_cpu(
    x: &[f32],
    w_x: &[f32],
    b: &[f32],
    batch: usize,
    d_in: usize,
    d_out: usize,
) -> Vec<f32> {
    let mut z = vec![0.0f32; batch * d_out];
    for i in 0..batch {
        for j in 0..d_out {
            let mut acc = b[j];
            for k in 0..d_in {
                acc += x[i * d_in + k] * w_x[k * d_out + j];
            }
            z[i * d_out + j] = acc.max(0.0);
        }
    }
    z
}

/// WGSL shader for fused ICNN linear layer with ReLU.
pub const ICNN_RELU_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w_x: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> z: array<f32>;

struct Uniforms {
    batch: u32,
    d_in: u32,
    d_out: u32,
};
@group(0) @binding(4) var<uniform> u: Uniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;
    if (row >= u.batch || col >= u.d_out) {
        return;
    }
    var acc: f32 = b[col];
    for (var k: u32 = 0u; k < u.d_in; k = k + 1u) {
        acc = acc + x[row * u.d_in + k] * w_x[k * u.d_out + col];
    }
    z[row * u.d_out + col] = max(0.0, acc);
}
"#;
