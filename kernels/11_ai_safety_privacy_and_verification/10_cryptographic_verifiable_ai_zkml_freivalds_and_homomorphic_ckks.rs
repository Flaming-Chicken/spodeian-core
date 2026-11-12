//! CPU (and optional WGSL) kernels for Cryptographic Verifiable AI ZKML Freivalds and Homomorphic CKKS.

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

/// WGSL compute shader for CKKS / CryptoNets polynomial activation:
/// y = a * x^2 + b * x + c.
pub const CKKS_DEGREE2_ACTIVATION_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> x_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> y_out: array<f32>;

struct Params {
    total_elements: u32,
    a: f32,
    b: f32,
    c: f32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.total_elements) {
        return;
    }

    let val = x_in[idx];
    y_out[idx] = params.a * val * val + params.b * val + params.c;
}
"#;

/// CPU reference implementation of CKKS degree-2 polynomial activation.
pub fn ckks_degree2_activation_cpu(x: &[f32], a: f32, b: f32, c: f32) -> Vec<f32> {
    x.iter().map(|&v| a * v * v + b * v + c).collect()
}
