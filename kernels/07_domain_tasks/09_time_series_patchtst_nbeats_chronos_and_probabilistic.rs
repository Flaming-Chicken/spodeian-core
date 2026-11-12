//! CPU and WGSL kernels for Time Series Patchtst Nbeats Chronos and Probabilistic.

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

/// Fused Reversible Instance Normalization (RevIN) forward kernel:
/// Computes per-instance mean, variance, and normalizes:
/// x: [B, M, L] row-major -> index = (b * M + m) * L + l
/// gamma, beta: [M]
/// Returns: (x_norm, means, stdevs)
pub fn revin_normalize_cpu(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    b: usize,
    m: usize,
    l: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let total_instances = b * m;
    let mut x_norm = vec![0.0f32; total_instances * l];
    let mut means = vec![0.0f32; total_instances];
    let mut stdevs = vec![0.0f32; total_instances];

    for bi in 0..b {
        for mi in 0..m {
            let inst_idx = bi * m + mi;
            let offset = inst_idx * l;
            let slice = &x[offset..offset + l];

            let sum: f32 = slice.iter().sum();
            let mean = sum / (l as f32);

            let var_sum: f32 = slice.iter().map(|&v| (v - mean) * (v - mean)).sum();
            let var = var_sum / (l as f32);
            let stdev = (var + eps).sqrt();

            means[inst_idx] = mean;
            stdevs[inst_idx] = stdev;

            let g = gamma[mi];
            let be = beta[mi];

            for li in 0..l {
                x_norm[offset + li] = ((slice[li] - mean) / stdev) * g + be;
            }
        }
    }

    (x_norm, means, stdevs)
}

/// Fused RevIN Normalization WGSL compute shader
pub const REVIN_NORMALIZE_WGSL: &str = r#"
struct Config {
    b: u32,
    m: u32,
    l: u32,
    eps: f32,
};

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> x_norm: array<f32>;
@group(0) @binding(4) var<storage, read_write> means: array<f32>;
@group(0) @binding(5) var<storage, read_write> stdevs: array<f32>;
@group(0) @binding(6) var<uniform> cfg: Config;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let inst_idx = global_id.x;
    let total_instances = cfg.b * cfg.m;
    if (inst_idx >= total_instances) {
        return;
    }

    let mi = inst_idx % cfg.m;
    let offset = inst_idx * cfg.l;

    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < cfg.l; i = i + 1u) {
        sum = sum + x[offset + i];
    }
    let mean: f32 = sum / f32(cfg.l);

    var var_sum: f32 = 0.0;
    for (var i: u32 = 0u; i < cfg.l; i = i + 1u) {
        let diff = x[offset + i] - mean;
        var_sum = var_sum + diff * diff;
    }
    let stdev: f32 = sqrt(var_sum / f32(cfg.l) + cfg.eps);

    means[inst_idx] = mean;
    stdevs[inst_idx] = stdev;

    let g = gamma[mi];
    let b = beta[mi];
    for (var i: u32 = 0u; i < cfg.l; i = i + 1u) {
        x_norm[offset + i] = ((x[offset + i] - mean) / stdev) * g + b;
    }
}
"#;

/// Dispatcher: fallback to CPU when GPU context is not available
pub fn revin_normalize_auto(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    b: usize,
    m: usize,
    l: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    revin_normalize_cpu(x, gamma, beta, b, m, l, eps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_revin_cpu() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0];
        let beta = vec![0.0];
        let (norm, means, stdevs) = revin_normalize_cpu(&x, &gamma, &beta, 1, 1, 4, 1e-5);
        assert_eq!(norm.len(), 4);
        assert_eq!(means.len(), 1);
        assert_eq!(stdevs.len(), 1);
        assert!((means[0] - 2.5).abs() < 1e-5);
    }
}
