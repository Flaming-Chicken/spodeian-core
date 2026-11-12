//! CPU (and optional WGSL) kernels for GANs Dcgan Wgan GP and Stylegan Concepts.

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

/// CPU kernel for Adaptive Instance Normalization: `y[b, c] = scale[b, c] * (x[b, c] - mean[b]) / sqrt(var[b] + eps) + bias[b, c]`.
pub fn adain_cpu(
    x: &[f32],
    scale: &[f32],
    bias: &[f32],
    batch: usize,
    channels: usize,
    eps: f32,
) -> Vec<f32> {
    let mut y = vec![0.0f32; batch * channels];
    for b in 0..batch {
        let offset = b * channels;
        let slice = &x[offset..offset + channels];
        let mean = slice.iter().sum::<f32>() / channels as f32;
        let var = slice.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / channels as f32;
        let inv_std = 1.0 / (var + eps).sqrt();

        for c in 0..channels {
            let idx = offset + c;
            y[idx] = (x[idx] - mean) * inv_std * scale[idx] + bias[idx];
        }
    }
    y
}

/// WGSL compute shader for fused Adaptive Instance Normalization (AdaIN).
pub const WGSL_ADAIN: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> channels: u32;
@group(0) @binding(5) var<uniform> eps: f32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.x;
    let offset = b * channels;
    if (offset >= arrayLength(&x)) {
        return;
    }
    var sum: f32 = 0.0;
    for (var c: u32 = 0u; c < channels; c = c + 1u) {
        sum = sum + x[offset + c];
    }
    let mean: f32 = sum / f32(channels);
    var var_sum: f32 = 0.0;
    for (var c: u32 = 0u; c < channels; c = c + 1u) {
        let diff = x[offset + c] - mean;
        var_sum = var_sum + diff * diff;
    }
    let inv_std: f32 = 1.0 / sqrt(var_sum / f32(channels) + eps);
    for (var c: u32 = 0u; c < channels; c = c + 1u) {
        let idx = offset + c;
        y[idx] = (x[idx] - mean) * inv_std * scale[idx] + bias[idx];
    }
}
"#;
