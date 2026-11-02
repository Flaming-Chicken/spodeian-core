//! CPU (and optional WGSL) kernels for Torch Fx Graph Capture and Custom Compiler Passes.

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

/// WGSL compute shader for algebraic Linear + BatchNorm parameter folding.
pub const ALGEBRAIC_LINEAR_BN_FOLD_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> w_in: array<f32>;
@group(0) @binding(1) var<storage, read> b_in: array<f32>;
@group(0) @binding(2) var<storage, read> mean_in: array<f32>;
@group(0) @binding(3) var<storage, read> var_in: array<f32>;
@group(0) @binding(4) var<storage, read> gamma_in: array<f32>;
@group(0) @binding(5) var<storage, read> beta_in: array<f32>;
@group(0) @binding(6) var<storage, read_write> w_fold_out: array<f32>;
@group(0) @binding(7) var<storage, read_write> b_fold_out: array<f32>;

struct Params {
    d_in: u32,
    d_out: u32,
    eps: f32,
    _pad: u32,
}
@group(0) @binding(8) var<uniform> params: Params;

@compute @workgroup_size(64)
fn fold_params(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total_weights = params.d_in * params.d_out;

    // Fold weights: idx runs from 0 to d_in * d_out - 1
    if (idx < total_weights) {
        let col = idx % params.d_out;
        let scale = gamma_in[col] / sqrt(var_in[col] + params.eps);
        w_fold_out[idx] = w_in[idx] * scale;
    }

    // Fold bias: idx runs from 0 to d_out - 1
    if (idx < params.d_out) {
        let scale = gamma_in[idx] / sqrt(var_in[idx] + params.eps);
        b_fold_out[idx] = (b_in[idx] - mean_in[idx]) * scale + beta_in[idx];
    }
}
"#;

/// CPU reference implementation for algebraic Linear + BatchNorm folding.
pub fn algebraic_linear_bn_fold_cpu(
    w: &[f32],
    b: &[f32],
    mean: &[f32],
    var: &[f32],
    gamma: &[f32],
    beta: &[f32],
    d_in: usize,
    d_out: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut w_fold = vec![0.0f32; d_in * d_out];
    let mut b_fold = vec![0.0f32; d_out];

    for o in 0..d_out {
        let scale = gamma[o] / (var[o] + eps).sqrt();
        b_fold[o] = (b[o] - mean[o]) * scale + beta[o];
        for i in 0..d_in {
            w_fold[i * d_out + o] = w[i * d_out + o] * scale;
        }
    }

    (w_fold, b_fold)
}
