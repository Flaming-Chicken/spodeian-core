//! CPU and WGSL kernels for Capstone Physics Smooth and Certified AI Pipeline.

/// WGSL compute shader for symplectic momentum update.
pub const FUSED_LEAPFROG_MOMENTUM_WGSL: &str = r#"
struct Uniforms {
    numel: u32,
    half_dt: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> p: array<f32>;
@group(0) @binding(2) var<storage, read> grad_v: array<f32>;
@group(0) @binding(3) var<storage, read_write> p_out: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn leapfrog_momentum_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= u.numel) { return; }

    p_out[idx] = p[idx] - u.half_dt * grad_v[idx];
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

/// CPU implementation of symplectic momentum update.
pub fn fused_leapfrog_momentum_cpu(p: &[f32], grad_v: &[f32], dt: f32) -> Vec<f32> {
    let half_dt = 0.5 * dt;
    p.iter()
        .zip(grad_v.iter())
        .map(|(&pv, &gv)| pv - half_dt * gv)
        .collect()
}

/// Dispatches symplectic momentum update with automatic CPU fallback.
pub fn fused_leapfrog_momentum_auto(p: &[f32], grad_v: &[f32], dt: f32) -> Vec<f32> {
    fused_leapfrog_momentum_cpu(p, grad_v, dt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fused_leapfrog_momentum() {
        let p = vec![1.0, 2.0];
        let grad = vec![4.0, -2.0];
        let p_out = fused_leapfrog_momentum_cpu(&p, &grad, 0.1);
        assert!((p_out[0] - 0.8).abs() < 1e-5);
        assert!((p_out[1] - 2.1).abs() < 1e-5);
    }
}
