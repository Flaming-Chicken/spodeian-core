//! CPU and WGSL kernels for ONNX Export Dynamo and ONNXRuntime Inference.

/// WGSL compute shader for fused bias addition + GELU activation in ONNX graph execution.
pub const FUSED_DYNAMO_GELU_BIAS_WGSL: &str = r#"
struct Uniforms {
    size: u32,
    d_model: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> x: array<f32>;        // [size]
@group(0) @binding(2) var<storage, read> bias: array<f32>;     // [d_model]
@group(0) @binding(3) var<storage, read_write> out: array<f32>; // [size]

@compute @workgroup_size(64, 1, 1)
fn dynamo_gelu_bias_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.size) { return; }

    let b_idx = i % u.d_model;
    let val = x[i] + bias[b_idx];

    // Standard erf-based GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    let sqrt_2_over_pi: f32 = 0.7978845608;
    let inner = sqrt_2_over_pi * (val + 0.044715 * val * val * val);
    let tanh_val = tanh(inner);
    out[i] = 0.5 * val * (1.0 + tanh_val);
}
"#;

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

/// CPU fused bias addition + GELU activation.
pub fn dynamo_gelu_bias_cpu(x: &[f32], bias: &[f32], d_model: usize) -> Vec<f32> {
    let sqrt_2_over_pi = 0.7978845608f32;
    x.iter()
        .enumerate()
        .map(|(i, &val)| {
            let b = bias[i % d_model];
            let v = val + b;
            let inner = sqrt_2_over_pi * (v + 0.044715 * v * v * v);
            0.5 * v * (1.0 + inner.tanh())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamo_gelu_bias_cpu() {
        let x = vec![0.0f32, 1.0f32];
        let bias = vec![0.0f32, 0.0f32];
        let out = dynamo_gelu_bias_cpu(&x, &bias, 2);
        assert!((out[0] - 0.0).abs() < 1e-4);
        assert!((out[1] - 0.8413).abs() < 1e-3);
    }
}
