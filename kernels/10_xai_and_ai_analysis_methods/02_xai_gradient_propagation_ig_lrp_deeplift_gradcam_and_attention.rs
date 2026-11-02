//! CPU (and optional WGSL) kernels for XAI Gradient Propagation IG LRP DeepLIFT Grad-CAM and Attention.

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

/// WGSL compute shader for Integrated Gradients path-integral Riemann sum accumulation.
pub const INTEGRATED_GRADIENTS_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> x_diff: array<f32>;
@group(0) @binding(1) var<storage, read> interpolated_grads: array<f32>;
@group(0) @binding(2) var<storage, read_write> ig_output: array<f32>;

struct Params {
    num_steps: u32,
    num_features: u32,
    _pad0: u32,
    _pad1: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.num_features) {
        return;
    }
    var sum_grad: f32 = 0.0;
    for (var m: u32 = 0u; m < params.num_steps; m = m + 1u) {
        sum_grad = sum_grad + interpolated_grads[m * params.num_features + i];
    }
    let avg_grad = sum_grad / f32(params.num_steps);
    ig_output[i] = x_diff[i] * avg_grad;
}
"#;

/// CPU reference implementation of Integrated Gradients path integral accumulation.
pub fn integrated_gradients_cpu(
    x: &[f32],
    x_baseline: &[f32],
    interpolated_grads: &[f32],
    num_steps: usize,
    num_features: usize,
) -> Vec<f32> {
    let mut ig = vec![0.0f32; num_features];
    let m_f = num_steps as f32;
    for i in 0..num_features {
        let diff = x[i] - x_baseline[i];
        let mut sum_g = 0.0f32;
        for m in 0..num_steps {
            sum_g += interpolated_grads[m * num_features + i];
        }
        ig[i] = diff * (sum_g / m_f);
    }
    ig
}
