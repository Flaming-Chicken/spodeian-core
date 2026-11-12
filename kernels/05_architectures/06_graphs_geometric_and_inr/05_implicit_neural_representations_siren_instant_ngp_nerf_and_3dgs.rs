//! CPU (and optional WGSL) kernels for Implicit Neural Representations: SIREN, Instant-NGP, NeRF, and 3DGS.

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

/// SIREN sinusoidal activation layer on CPU: y = sin(omega_0 * (x @ W + b)).
pub fn siren_layer_cpu(
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    batch: usize,
    d_in: usize,
    d_out: usize,
    omega_0: f32,
) -> Vec<f32> {
    let mut y = vec![0.0f32; batch * d_out];
    for b in 0..batch {
        for o in 0..d_out {
            let mut dot = bias[o];
            for i in 0..d_in {
                dot += x[b * d_in + i] * w[i * d_out + o];
            }
            y[b * d_out + o] = (omega_0 * dot).sin();
        }
    }
    y
}

/// 3DGS 2D projected Gaussian alpha evaluation on CPU.
pub fn gaussian_splat_2d_eval_cpu(
    x: &[f32],
    mu: &[f32],
    cov_inv: &[f32],
    batch: usize,
) -> Vec<f32> {
    let mut alphas = Vec::with_capacity(batch);
    for b in 0..batch {
        let dx = x[b * 2 + 0] - mu[0];
        let dy = x[b * 2 + 1] - mu[1];

        let px = dx * cov_inv[0 * 2 + 0] + dy * cov_inv[1 * 2 + 0];
        let py = dx * cov_inv[0 * 2 + 1] + dy * cov_inv[1 * 2 + 1];

        let mahalanobis = px * dx + py * dy;
        alphas.push((-0.5 * mahalanobis).exp());
    }
    alphas
}

/// WGSL shader for parallel SIREN layer matrix multiplication and sinusoidal activation.
pub const WGSL_SIREN_LAYER: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

struct Uniforms {
    batch_size: u32,
    d_in: u32,
    d_out: u32,
    omega_0: f32,
};
@group(0) @binding(4) var<uniform> u: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = u.batch_size * u.d_out;
    if (idx >= total) {
        return;
    }
    let b = idx / u.d_out;
    let o = idx % u.d_out;

    var dot: f32 = bias[o];
    for (var i: u32 = 0u; i < u.d_in; i = i + 1u) {
        dot = dot + x[b * u.d_in + i] * w[i * u.d_out + o];
    }
    y[idx] = sin(u.omega_0 * dot);
}
"#;
