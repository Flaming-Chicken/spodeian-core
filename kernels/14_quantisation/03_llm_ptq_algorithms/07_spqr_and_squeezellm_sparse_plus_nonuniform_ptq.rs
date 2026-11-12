//! CPU and WGSL kernels for SpQR and SqueezeLLM: Sparse Plus Nonuniform PTQ.

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

/// WGSL compute shader for fused SpQR sparse + dense GEMM:
/// `Y[b, o] = bias[o] + sum_i X[b, i] * (W_dense[i, o] + W_sparse[i, o])`
pub const SPQR_FUSED_GEMM_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w_dense: array<f32>;
@group(0) @binding(2) var<storage, read> w_sparse: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> y_out: array<f32>;

struct Uniforms {
    batch: u32,
    d_in: u32,
    d_out: u32,
};
@group(0) @binding(5) var<uniform> params: Uniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.x;
    let o = global_id.y;
    if (b >= params.batch || o >= params.d_out) {
        return;
    }
    var acc: f32 = bias[o];
    for (var i: u32 = 0u; i < params.d_in; i = i + 1u) {
        let x_val = x[b * params.d_in + i];
        let w_d = w_dense[i * params.d_out + o];
        let w_s = w_sparse[i * params.d_out + o];
        acc = acc + x_val * (w_d + w_s);
    }
    y_out[b * params.d_out + o] = acc;
}
"#;

/// CPU reference implementation of fused SpQR forward computation:
/// Computes Y = X @ (W_dense + W_sparse) + bias
pub fn spqr_forward_cpu(
    x: &[f32],
    w_dense: &[f32],
    w_sparse: &[f32],
    bias: &[f32],
    batch: usize,
    d_in: usize,
    d_out: usize,
) -> Vec<f32> {
    let mut y = vec![0.0f32; batch * d_out];
    for b in 0..batch {
        for o in 0..d_out {
            let mut acc = bias[o];
            for i in 0..d_in {
                let x_val = x[b * d_in + i];
                let wd = w_dense[i * d_out + o];
                let ws = w_sparse[i * d_out + o];
                acc += x_val * (wd + ws);
            }
            y[b * d_out + o] = acc;
        }
    }
    y
}

/// Auto dispatcher: returns CPU reference fallback.
pub fn spqr_forward_auto(
    x: &[f32],
    w_dense: &[f32],
    w_sparse: &[f32],
    bias: &[f32],
    batch: usize,
    d_in: usize,
    d_out: usize,
) -> Vec<f32> {
    spqr_forward_cpu(x, w_dense, w_sparse, bias, batch, d_in, d_out)
}
