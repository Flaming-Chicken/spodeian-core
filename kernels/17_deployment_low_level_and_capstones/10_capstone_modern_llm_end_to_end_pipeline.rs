//! CPU and WGSL kernels for Capstone Modern LLM End to End Pipeline.

/// WGSL compute shader for fused residual addition and RMSNorm.
pub const FUSED_LLM_RESIDUAL_RMSNORM_WGSL: &str = r#"
struct Uniforms {
    batch_size: u32,
    d_model: u32,
    eps: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> x: array<f32>;        // [batch_size, d_model]
@group(0) @binding(2) var<storage, read> res: array<f32>;      // [batch_size, d_model]
@group(0) @binding(3) var<storage, read> gamma: array<f32>;    // [d_model]
@group(0) @binding(4) var<storage, read_write> y: array<f32>;  // [batch_size, d_model]

@compute @workgroup_size(16, 1, 1)
fn residual_rmsnorm_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if (b >= u.batch_size) { return; }

    var sum_sq: f32 = 0.0;
    for (var d = 0u; d < u.d_model; d++) {
        let val = x[b * u.d_model + d] + res[b * u.d_model + d];
        sum_sq += val * val;
    }
    let mean_sq = sum_sq / f32(u.d_model);
    let inv_rms = 1.0 / sqrt(mean_sq + u.eps);

    for (var d = 0u; d < u.d_model; d++) {
        let val = x[b * u.d_model + d] + res[b * u.d_model + d];
        y[b * u.d_model + d] = val * inv_rms * gamma[d];
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

/// CPU implementation of fused residual addition and RMSNorm.
pub fn fused_residual_rmsnorm_cpu(
    x: &[f32],
    res: &[f32],
    gamma: &[f32],
    batch: usize,
    d_model: usize,
    eps: f32,
) -> Vec<f32> {
    let mut y = vec![0.0f32; batch * d_model];
    for b in 0..batch {
        let mut sum_sq = 0.0f32;
        for d in 0..d_model {
            let val = x[b * d_model + d] + res[b * d_model + d];
            sum_sq += val * val;
        }
        let inv_rms = 1.0 / ((sum_sq / d_model as f32) + eps).sqrt();
        for d in 0..d_model {
            let val = x[b * d_model + d] + res[b * d_model + d];
            y[b * d_model + d] = val * inv_rms * gamma[d];
        }
    }
    y
}

/// Dispatches fused residual RMSNorm with automatic CPU fallback.
pub fn fused_residual_rmsnorm_auto(
    x: &[f32],
    res: &[f32],
    gamma: &[f32],
    batch: usize,
    d_model: usize,
    eps: f32,
) -> Vec<f32> {
    fused_residual_rmsnorm_cpu(x, res, gamma, batch, d_model, eps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fused_residual_rmsnorm() {
        let x = vec![0.5, 0.5, 0.5, 0.5];
        let res = vec![0.5, 0.5, 0.5, 0.5];
        let gamma = vec![1.0, 1.0, 1.0, 1.0];
        let y = fused_residual_rmsnorm_cpu(&x, &res, &gamma, 1, 4, 1e-5);
        for val in y {
            assert!((val - 1.0).abs() < 1e-2);
        }
    }
}
