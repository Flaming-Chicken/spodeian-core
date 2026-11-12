//! CPU (and optional WGSL) kernels for Causal Representation Identifiability iVAE Synthetic Control and Symbolic Sr.

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

/// WGSL compute shader for iVAE conditional Gaussian KL divergence reduction.
pub const IVAE_KL_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> mu_z: array<f32>;
@group(0) @binding(1) var<storage, read> logvar_z: array<f32>;
@group(0) @binding(2) var<storage, read> mu_u: array<f32>;
@group(0) @binding(3) var<storage, read> logvar_u: array<f32>;
@group(0) @binding(4) var<storage, read_write> kl_out: array<f32>;

struct Params {
    batch_size: u32,
    latent_dim: u32,
    _pad0: u32,
    _pad1: u32,
}
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if (b >= params.batch_size) {
        return;
    }
    var sum_kl: f32 = 0.0;
    for (var j: u32 = 0u; j < params.latent_dim; j = j + 1u) {
        let idx = b * params.latent_dim + j;
        let lz = logvar_z[idx];
        let lu = logvar_u[idx];
        let vz = exp(lz);
        let vu = exp(lu);
        let dmu = mu_z[idx] - mu_u[idx];
        let term = (vz + dmu * dmu) / vu - 1.0 - (lz - lu);
        sum_kl = sum_kl + 0.5 * term;
    }
    kl_out[b] = sum_kl;
}
"#;

/// CPU reference implementation of iVAE conditional Gaussian KL divergence.
pub fn ivae_kl_cpu(
    mu_z: &[f32],
    logvar_z: &[f32],
    mu_u: &[f32],
    logvar_u: &[f32],
    batch_size: usize,
    latent_dim: usize,
) -> Vec<f32> {
    let mut kl_out = vec![0.0f32; batch_size];
    for b in 0..batch_size {
        let mut sum_kl = 0.0f32;
        for j in 0..latent_dim {
            let idx = b * latent_dim + j;
            let lz = logvar_z[idx];
            let lu = logvar_u[idx];
            let vz = lz.exp();
            let vu = lu.exp();
            let dmu = mu_z[idx] - mu_u[idx];
            let term = (vz + dmu * dmu) / vu - 1.0 - (lz - lu);
            sum_kl += 0.5 * term;
        }
        kl_out[b] = sum_kl;
    }
    kl_out
}

/// Auto-dispatching iVAE conditional Gaussian KL divergence (CPU fallback).
pub fn ivae_kl_auto(
    mu_z: &[f32],
    logvar_z: &[f32],
    mu_u: &[f32],
    logvar_u: &[f32],
    batch_size: usize,
    latent_dim: usize,
) -> Vec<f32> {
    ivae_kl_cpu(mu_z, logvar_z, mu_u, logvar_u, batch_size, latent_dim)
}
