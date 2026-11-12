//! CPU (and optional WGSL) kernels for Normalizing Flows Realnvp Glow and Spline Flows.

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

/// CPU kernel for RealNVP affine coupling forward step: `y_b = x_b * exp(s) + t`.
pub fn realnvp_coupling_cpu(x_b: &[f32], s: &[f32], t: &[f32]) -> Vec<f32> {
    x_b.iter()
        .zip(s.iter())
        .zip(t.iter())
        .map(|((&xb, &si), &ti)| xb * si.exp() + ti)
        .collect()
}

/// WGSL compute shader for fused RealNVP affine coupling transform.
pub const WGSL_REALNVP: &str = r#"
@group(0) @binding(0) var<storage, read> x_b: array<f32>;
@group(0) @binding(1) var<storage, read> s: array<f32>;
@group(0) @binding(2) var<storage, read> t: array<f32>;
@group(0) @binding(3) var<storage, read_write> y_b: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&x_b)) {
        return;
    }
    y_b[idx] = x_b[idx] * exp(s[idx]) + t[idx];
}
"#;
