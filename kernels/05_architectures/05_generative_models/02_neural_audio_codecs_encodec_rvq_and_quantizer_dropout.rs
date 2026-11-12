//! CPU (and optional WGSL) kernels for Neural Audio Codecs Encodec RVQ and Quantizer Dropout.

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

/// CPU kernel for Snake periodic activation: `y = x + (1 / alpha) * sin^2(alpha * x)`.
pub fn snake_activation_cpu(x: &[f32], alpha: f32) -> Vec<f32> {
    let inv_alpha = 1.0 / alpha;
    x.iter()
        .map(|&v| {
            let sin_val = (alpha * v).sin();
            v + inv_alpha * sin_val * sin_val
        })
        .collect()
}

/// WGSL compute shader for fused Snake periodic activation.
pub const WGSL_SNAKE: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> alpha: f32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&x)) {
        return;
    }
    let val = x[idx];
    let sin_val = sin(alpha * val);
    y[idx] = val + (1.0 / alpha) * sin_val * sin_val;
}
"#;
