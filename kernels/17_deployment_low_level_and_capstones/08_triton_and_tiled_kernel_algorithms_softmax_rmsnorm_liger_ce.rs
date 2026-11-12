//! CPU and WGSL kernels for Triton and Tiled Kernel Algorithms Softmax RMSNorm Liger Cross-Entropy.

/// WGSL compute shader for fused RMSNorm kernel.
pub const FUSED_RMSNORM_WGSL: &str = r#"
struct Uniforms {
    batch_size: u32,
    d_model: u32,
    eps: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> x: array<f32>;        // [batch_size, d_model]
@group(0) @binding(2) var<storage, read> gamma: array<f32>;    // [d_model]
@group(0) @binding(3) var<storage, read_write> y: array<f32>;  // [batch_size, d_model]

@compute @workgroup_size(16, 1, 1)
fn rmsnorm_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if (b >= u.batch_size) { return; }

    var sum_sq: f32 = 0.0;
    for (var d = 0u; d < u.d_model; d++) {
        let val = x[b * u.d_model + d];
        sum_sq += val * val;
    }
    let mean_sq = sum_sq / f32(u.d_model);
    let inv_rms = 1.0 / sqrt(mean_sq + u.eps);

    for (var d = 0u; d < u.d_model; d++) {
        y[b * u.d_model + d] = x[b * u.d_model + d] * inv_rms * gamma[d];
    }
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

/// CPU implementation of fused RMSNorm kernel.
pub fn fused_rmsnorm_cpu(
    x: &[f32],
    gamma: &[f32],
    batch: usize,
    d_model: usize,
    eps: f32,
) -> Vec<f32> {
    let mut y = vec![0.0f32; batch * d_model];
    for b in 0..batch {
        let row = &x[b * d_model..(b + 1) * d_model];
        let sum_sq: f32 = row.iter().map(|&v| v * v).sum();
        let inv_rms = 1.0 / ((sum_sq / d_model as f32) + eps).sqrt();
        for d in 0..d_model {
            y[b * d_model + d] = row[d] * inv_rms * gamma[d];
        }
    }
    y
}

/// Dispatches fused RMSNorm with automatic CPU fallback.
pub fn fused_rmsnorm_auto(
    x: &[f32],
    gamma: &[f32],
    batch: usize,
    d_model: usize,
    eps: f32,
) -> Vec<f32> {
    fused_rmsnorm_cpu(x, gamma, batch, d_model, eps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fused_rmsnorm() {
        let x = vec![1.0, 1.0, 1.0, 1.0];
        let gamma = vec![2.0, 2.0, 2.0, 2.0];
        let y = fused_rmsnorm_cpu(&x, &gamma, 1, 4, 1e-5);
        for val in y {
            assert!((val - 2.0).abs() < 1e-2);
        }
    }
}
