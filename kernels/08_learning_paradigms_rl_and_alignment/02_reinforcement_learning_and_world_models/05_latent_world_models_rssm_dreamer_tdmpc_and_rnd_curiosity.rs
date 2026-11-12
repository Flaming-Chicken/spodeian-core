//! CPU and WGSL kernels for Latent World Models Rssm Dreamer Tdmpc and Rnd Curiosity.

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

/// Gaussian KL divergence CPU kernel:
/// D_KL(q || p) = 0.5 * sum_d [ (var_q + (mu_q - mu_p)^2) / var_p - 1 + 2*(log_std_p - log_std_q) ]
pub fn gaussian_kl_divergence_cpu(
    mu_q: &[f32],
    log_std_q: &[f32],
    mu_p: &[f32],
    log_std_p: &[f32],
    b: usize,
    d: usize,
) -> Vec<f32> {
    let mut kl = vec![0.0f32; b];
    for bi in 0..b {
        let offset = bi * d;
        let mut sum_kl = 0.0f32;
        for di in 0..d {
            let idx = offset + di;
            let var_q = (2.0f32 * log_std_q[idx]).exp();
            let var_p = (2.0f32 * log_std_p[idx]).exp();
            let diff = mu_q[idx] - mu_p[idx];
            let term =
                (var_q + diff * diff) / var_p - 1.0f32 + 2.0f32 * (log_std_p[idx] - log_std_q[idx]);
            sum_kl += 0.5f32 * term;
        }
        kl[bi] = sum_kl;
    }
    kl
}

/// Gaussian KL divergence WGSL compute shader
pub const GAUSSIAN_KL_DIVERGENCE_WGSL: &str = r#"
struct Config {
    b: u32,
    d: u32,
};

@group(0) @binding(0) var<storage, read> mu_q: array<f32>;
@group(0) @binding(1) var<storage, read> log_std_q: array<f32>;
@group(0) @binding(2) var<storage, read> mu_p: array<f32>;
@group(0) @binding(3) var<storage, read> log_std_p: array<f32>;
@group(0) @binding(4) var<storage, read_write> kl_out: array<f32>;
@group(0) @binding(5) var<uniform> cfg: Config;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let bi = global_id.x;
    if (bi >= cfg.b) {
        return;
    }

    let offset = bi * cfg.d;
    var sum_kl: f32 = 0.0;

    for (var di: u32 = 0u; di < cfg.d; di = di + 1u) {
        let idx = offset + di;
        let var_q = exp(2.0 * log_std_q[idx]);
        let var_p = exp(2.0 * log_std_p[idx]);
        let diff = mu_q[idx] - mu_p[idx];
        let term = (var_q + diff * diff) / var_p - 1.0 + 2.0 * (log_std_p[idx] - log_std_q[idx]);
        sum_kl = sum_kl + 0.5 * term;
    }

    kl_out[bi] = sum_kl;
}
"#;
