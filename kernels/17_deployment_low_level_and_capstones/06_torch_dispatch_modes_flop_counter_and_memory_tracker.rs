//! CPU and WGSL kernels for Torch Dispatch Modes FLOP Counter and Memory Tracker.

/// WGSL compute shader for fused residual addition and SiLU activation.
pub const FUSED_RESIDUAL_SILU_WGSL: &str = r#"
struct Uniforms {
    numel: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> z: array<f32>;        // intermediate activations
@group(0) @binding(2) var<storage, read> residual: array<f32>; // residual input
@group(0) @binding(3) var<storage, read_write> y: array<f32>;  // output

@compute @workgroup_size(64, 1, 1)
fn residual_silu_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= u.numel) { return; }

    let val = z[idx];
    let sigmoid_val = 1.0 / (1.0 + exp(-val));
    let silu_val = val * sigmoid_val;
    y[idx] = silu_val + residual[idx];
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

/// CPU implementation of fused residual addition and SiLU activation.
pub fn fused_residual_silu_cpu(z: &[f32], residual: &[f32]) -> Vec<f32> {
    z.iter()
        .zip(residual.iter())
        .map(|(&zv, &rv)| {
            let sig = 1.0 / (1.0 + (-zv).exp());
            (zv * sig) + rv
        })
        .collect()
}

/// Dispatches fused residual SiLU with automatic CPU fallback.
pub fn fused_residual_silu_auto(z: &[f32], residual: &[f32]) -> Vec<f32> {
    fused_residual_silu_cpu(z, residual)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fused_residual_silu() {
        let z = vec![0.0, 1.0, -1.0];
        let r = vec![1.0, 2.0, 3.0];
        let out = fused_residual_silu_cpu(&z, &r);
        assert!((out[0] - 1.0).abs() < 1e-5);
        let expected1 = 1.0 * (1.0 / (1.0 + (-1.0f32).exp())) + 2.0;
        assert!((out[1] - expected1).abs() < 1e-5);
    }
}
