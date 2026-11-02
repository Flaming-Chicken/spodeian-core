//! CPU and WGSL kernels for PT2E Export Quantization Workflow (Conv-BN Folding and Q/DQ).

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

/// WGSL compute shader for algebraic Conv2d + BatchNorm parameter fusion.
pub const CONV2D_BN_FOLD_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> w_conv: array<f32>;
@group(0) @binding(1) var<storage, read> b_conv: array<f32>;
@group(0) @binding(2) var<storage, read> bn_mean: array<f32>;
@group(0) @binding(3) var<storage, read> bn_var: array<f32>;
@group(0) @binding(4) var<storage, read> bn_gamma: array<f32>;
@group(0) @binding(5) var<storage, read> bn_beta: array<f32>;

@group(0) @binding(6) var<storage, read_write> w_folded: array<f32>;
@group(0) @binding(7) var<storage, read_write> b_folded: array<f32>;

struct Uniforms {
    c_out: u32,
    c_in: u32,
    k_sq: u32,
    eps: f32,
};
@group(0) @binding(8) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_weights = params.c_out * params.c_in * params.k_sq;

    if (idx < params.c_out) {
        let denom = sqrt(bn_var[idx] + params.eps);
        let scale = bn_gamma[idx] / denom;
        b_folded[idx] = (b_conv[idx] - bn_mean[idx]) * scale + bn_beta[idx];
    }

    if (idx < total_weights) {
        let c_out_idx = idx / (params.c_in * params.k_sq);
        let denom = sqrt(bn_var[c_out_idx] + params.eps);
        let scale = bn_gamma[c_out_idx] / denom;
        w_folded[idx] = w_conv[idx] * scale;
    }
}
"#;

/// CPU reference implementation of Conv2d + BatchNorm parameter folding.
pub fn conv2d_bn_fold_cpu(
    w_conv: &[f32],
    b_conv: &[f32],
    bn_mean: &[f32],
    bn_var: &[f32],
    bn_gamma: &[f32],
    bn_beta: &[f32],
    c_out: usize,
    c_in: usize,
    k_sq: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut w_folded = vec![0.0f32; w_conv.len()];
    let mut b_folded = vec![0.0f32; c_out];

    for c in 0..c_out {
        let denom = (bn_var[c] + eps).sqrt();
        let scale = bn_gamma[c] / denom;
        b_folded[c] = (b_conv[c] - bn_mean[c]) * scale + bn_beta[c];

        let start = c * c_in * k_sq;
        let end = (c + 1) * c_in * k_sq;
        for i in start..end {
            w_folded[i] = w_conv[i] * scale;
        }
    }

    (w_folded, b_folded)
}

/// Auto dispatcher: returns CPU reference fallback.
pub fn conv2d_bn_fold_auto(
    w_conv: &[f32],
    b_conv: &[f32],
    bn_mean: &[f32],
    bn_var: &[f32],
    bn_gamma: &[f32],
    bn_beta: &[f32],
    c_out: usize,
    c_in: usize,
    k_sq: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    conv2d_bn_fold_cpu(
        w_conv, b_conv, bn_mean, bn_var, bn_gamma, bn_beta, c_out, c_in, k_sq, eps,
    )
}
