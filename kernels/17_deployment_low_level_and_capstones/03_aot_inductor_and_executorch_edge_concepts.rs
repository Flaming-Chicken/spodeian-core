//! CPU and WGSL kernels for AOT Inductor and ExecuTorch Edge Concepts.

/// WGSL compute shader for fused LayerNorm + SiLU activation on pre-allocated static arenas.
pub const FUSED_AOT_LAYERNORM_SILU_WGSL: &str = r#"
struct Uniforms {
    batch_size: u32,
    d_model: u32,
    eps: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> z: array<f32>;        // [batch_size, d_model]
@group(0) @binding(2) var<storage, read> gamma: array<f32>;    // [d_model]
@group(0) @binding(3) var<storage, read> beta: array<f32>;     // [d_model]
@group(0) @binding(4) var<storage, read_write> y: array<f32>;  // [batch_size, d_model]

@compute @workgroup_size(16, 1, 1)
fn aot_layernorm_silu_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if (b >= u.batch_size) { return; }

    var sum: f32 = 0.0;
    for (var d = 0u; d < u.d_model; d++) {
        sum += z[b * u.d_model + d];
    }
    let mean = sum / f32(u.d_model);

    var sq_sum: f32 = 0.0;
    for (var d = 0u; d < u.d_model; d++) {
        let diff = z[b * u.d_model + d] - mean;
        sq_sum += diff * diff;
    }
    let var_val = sq_sum / f32(u.d_model);
    let inv_std = 1.0 / sqrt(var_val + u.eps);

    for (var d = 0u; d < u.d_model; d++) {
        let norm_val = (z[b * u.d_model + d] - mean) * inv_std * gamma[d] + beta[d];
        let sigmoid_val = 1.0 / (1.0 + exp(-norm_val));
        y[b * u.d_model + d] = norm_val * sigmoid_val;
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

/// CPU fused LayerNorm + SiLU kernel.
pub fn aot_layernorm_silu_cpu(
    z: &[f32],
    gamma: &[f32],
    beta: &[f32],
    batch: usize,
    d_model: usize,
) -> Vec<f32> {
    let mut y = vec![0.0f32; batch * d_model];

    for b in 0..batch {
        let row = &z[b * d_model..(b + 1) * d_model];
        let mean = row.iter().sum::<f32>() / (d_model as f32);
        let var = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / (d_model as f32);
        let inv_std = 1.0 / (var + 1e-5).sqrt();

        for d in 0..d_model {
            let norm = (row[d] - mean) * inv_std * gamma[d] + beta[d];
            let sig = 1.0 / (1.0 + (-norm).exp());
            y[b * d_model + d] = norm * sig;
        }
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aot_layernorm_silu_cpu() {
        let z = vec![1.0, 2.0, 3.0, 4.0]; // 1 row, 4 dims
        let gamma = vec![1.0, 1.0, 1.0, 1.0];
        let beta = vec![0.0, 0.0, 0.0, 0.0];
        let out = aot_layernorm_silu_cpu(&z, &gamma, &beta, 1, 4);
        assert_eq!(out.len(), 4);
        assert!(out[3] > out[0]); // monotonic with input
    }
}
