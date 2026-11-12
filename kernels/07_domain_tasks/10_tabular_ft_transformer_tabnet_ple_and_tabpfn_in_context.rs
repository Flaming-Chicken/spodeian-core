//! CPU and WGSL kernels for Tabular Ft Transformer Tabnet Ple and Tabpfn In Context.

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

/// Fused FT-Transformer Numerical Feature Tokenization:
/// token[b, j, d] = x_num[b, j] * w_num[j, d] + b_num[j, d]
/// x_num: [B, N]
/// w_num: [N, D]
/// b_num: [N, D]
/// out: [B, N, D]
pub fn ft_numerical_tokenize_cpu(
    x_num: &[f32],
    w_num: &[f32],
    b_num: &[f32],
    b: usize,
    n: usize,
    d: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; b * n * d];
    for bi in 0..b {
        for ni in 0..n {
            let x_val = x_num[bi * n + ni];
            for di in 0..d {
                let w_val = w_num[ni * d + di];
                let b_val = b_num[ni * d + di];
                out[(bi * n + ni) * d + di] = x_val * w_val + b_val;
            }
        }
    }
    out
}

/// Fused FT-Transformer Numerical Feature Tokenization WGSL compute shader
pub const FT_NUMERICAL_TOKENIZE_WGSL: &str = r#"
struct Config {
    b: u32,
    n: u32,
    d: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> x_num: array<f32>;
@group(0) @binding(1) var<storage, read> w_num: array<f32>;
@group(0) @binding(2) var<storage, read> b_num: array<f32>;
@group(0) @binding(3) var<storage, read_write> tokens: array<f32>;
@group(0) @binding(4) var<uniform> cfg: Config;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let di = global_id.x;
    let ni = global_id.y;
    let bi = global_id.z;

    if (bi >= cfg.b || ni >= cfg.n || di >= cfg.d) {
        return;
    }

    let x_val = x_num[bi * cfg.n + ni];
    let w_val = w_num[ni * cfg.d + di];
    let b_val = b_num[ni * cfg.d + di];

    let out_idx = (bi * cfg.n + ni) * cfg.d + di;
    tokens[out_idx] = x_val * w_val + b_val;
}
"#;

/// Dispatcher: fallback to CPU when GPU context is not available
pub fn ft_numerical_tokenize_auto(
    x_num: &[f32],
    w_num: &[f32],
    b_num: &[f32],
    b: usize,
    n: usize,
    d: usize,
) -> Vec<f32> {
    ft_numerical_tokenize_cpu(x_num, w_num, b_num, b, n, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ft_numerical_tokenize_cpu() {
        let x = vec![2.0, 3.0]; // [1, 2]
        let w = vec![1.0, 0.5, 2.0, -1.0]; // [2, 2]
        let b = vec![0.1, 0.2, 0.3, 0.4]; // [2, 2]
        let res = ft_numerical_tokenize_cpu(&x, &w, &b, 1, 2, 2);
        assert_eq!(res.len(), 4);
        assert!((res[0] - (2.0 * 1.0 + 0.1)).abs() < 1e-5);
    }
}
