//! CPU (and optional WGSL) kernels for Diffusion DDPM Ddim Edm and DiT.

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

/// CPU kernel for DDPM forward diffusion: `x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * noise`.
pub fn ddpm_forward_diffuse_cpu(x_0: &[f32], noise: &[f32], alpha_bar: f32) -> Vec<f32> {
    let sqrt_ab = alpha_bar.sqrt();
    let sqrt_one_minus = (1.0 - alpha_bar).sqrt();
    x_0.iter()
        .zip(noise.iter())
        .map(|(&x, &eps)| sqrt_ab * x + sqrt_one_minus * eps)
        .collect()
}

/// WGSL compute shader for fused DDPM forward diffusion step.
pub const WGSL_DDPM: &str = r#"
@group(0) @binding(0) var<storage, read> x_0: array<f32>;
@group(0) @binding(1) var<storage, read> noise: array<f32>;
@group(0) @binding(2) var<storage, read_write> x_t: array<f32>;
@group(0) @binding(3) var<uniform> alpha_bar: f32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&x_0)) {
        return;
    }
    let sqrt_ab = sqrt(alpha_bar);
    let sqrt_one_minus = sqrt(1.0 - alpha_bar);
    x_t[idx] = sqrt_ab * x_0[idx] + sqrt_one_minus * noise[idx];
}
"#;
