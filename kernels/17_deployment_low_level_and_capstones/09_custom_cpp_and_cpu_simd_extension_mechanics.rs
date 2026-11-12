//! CPU and WGSL kernels for Custom C++ and CPU SIMD Extension Mechanics.

/// WGSL compute shader for fused vector FMA and LeakyReLU activation.
pub const FUSED_SIMD_FMA_LEAKY_RELU_WGSL: &str = r#"
struct Uniforms {
    numel: u32,
    negative_slope: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn simd_fma_leaky_relu_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= u.numel) { return; }

    let z = x[idx] * w[idx] + b[idx];
    y[idx] = select(z * u.negative_slope, z, z >= 0.0);
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

/// CPU implementation of fused vector FMA and LeakyReLU.
pub fn fused_simd_fma_leaky_relu_cpu(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    negative_slope: f32,
) -> Vec<f32> {
    x.iter()
        .zip(w.iter())
        .zip(b.iter())
        .map(|((&xi, &wi), &bi)| {
            let z = xi * wi + bi;
            if z >= 0.0 {
                z
            } else {
                z * negative_slope
            }
        })
        .collect()
}

/// Dispatches fused SIMD FMA LeakyReLU with automatic CPU fallback.
pub fn fused_simd_fma_leaky_relu_auto(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    negative_slope: f32,
) -> Vec<f32> {
    fused_simd_fma_leaky_relu_cpu(x, w, b, negative_slope)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fused_simd_fma_leaky_relu() {
        let x = vec![2.0, -2.0];
        let w = vec![3.0, 3.0];
        let b = vec![1.0, 1.0];
        let y = fused_simd_fma_leaky_relu_cpu(&x, &w, &b, 0.1);
        assert!((y[0] - 7.0).abs() < 1e-5);
        assert!((y[1] - (-0.5)).abs() < 1e-5);
    }
}
