//! CPU and WGSL kernels for Custom Tensor Subclasses Quantized and Tracked Tensors.

/// WGSL compute shader for fused dequantization affine operation.
pub const FUSED_DEQUANT_AFFINE_WGSL: &str = r#"
struct Uniforms {
    numel: u32,
    scale: f32,
    zero_point: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> int_w: array<f32>;    // stored as float-cast ints
@group(0) @binding(2) var<storage, read_write> fp_w: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn dequant_affine_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= u.numel) { return; }

    fp_w[idx] = (int_w[idx] - u.zero_point) * u.scale;
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

/// CPU implementation of fused dequantization affine operation.
pub fn dequant_affine_cpu(int_w: &[f32], scale: f32, zero_point: f32) -> Vec<f32> {
    int_w.iter().map(|&w| (w - zero_point) * scale).collect()
}

/// Dispatches dequant affine kernel with automatic CPU fallback.
pub fn dequant_affine_auto(int_w: &[f32], scale: f32, zero_point: f32) -> Vec<f32> {
    dequant_affine_cpu(int_w, scale, zero_point)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dequant_affine() {
        let int_w = vec![10.0, -10.0, 0.0];
        let fp_w = dequant_affine_cpu(&int_w, 0.5, 0.0);
        assert!((fp_w[0] - 5.0).abs() < 1e-5);
        assert!((fp_w[1] - (-5.0)).abs() < 1e-5);
        assert!((fp_w[2] - 0.0).abs() < 1e-5);
    }
}
