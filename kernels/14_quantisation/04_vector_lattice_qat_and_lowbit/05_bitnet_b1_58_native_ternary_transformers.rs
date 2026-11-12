//! CPU and WGSL kernels for BitNet b1.58 Native Ternary Transformers.

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

/// WGSL compute shader for fused BitLinear integer-ternary matmul with token-wise scaling:
/// `Y[b, o] = bias[o] + ((gamma[b] * beta) / 127.0) * sum_i (x_int8[b, i] * w_ternary[i, o])`
pub const BITLINEAR_INT8_TERNARY_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> x_int8: array<f32>;
@group(0) @binding(1) var<storage, read> w_ternary: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out_y: array<f32>;

struct Uniforms {
    batch: u32,
    d_in: u32,
    d_out: u32,
    beta: f32,
};
@group(0) @binding(5) var<uniform> params: Uniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.x;
    let o = global_id.y;
    if (b >= params.batch || o >= params.d_out) {
        return;
    }
    var acc: f32 = 0.0;
    for (var i: u32 = 0u; i < params.d_in; i = i + 1u) {
        let x_val = x_int8[b * params.d_in + i];
        let w_val = w_ternary[i * params.d_out + o];
        acc = acc + x_val * w_val;
    }
    let scale = (gamma[b] * params.beta) / 127.0;
    let idx = b * params.d_out + o;
    out_y[idx] = bias[o] + acc * scale;
}
"#;

/// CPU reference implementation of fused BitLinear INT8 x Ternary forward GEMM.
pub fn bitlinear_forward_cpu(
    x_int8: &[f32],
    w_ternary: &[f32],
    gamma: &[f32],
    beta: f32,
    bias: &[f32],
    batch: usize,
    d_in: usize,
    d_out: usize,
) -> Vec<f32> {
    let mut out_y = vec![0.0f32; batch * d_out];
    for b in 0..batch {
        let scale = (gamma[b] * beta) / 127.0;
        for o in 0..d_out {
            let mut acc = 0.0f32;
            for i in 0..d_in {
                acc += x_int8[b * d_in + i] * w_ternary[i * d_out + o];
            }
            out_y[b * d_out + o] = bias[o] + acc * scale;
        }
    }
    out_y
}

/// Auto dispatcher: returns CPU reference fallback.
pub fn bitlinear_forward_auto(
    x_int8: &[f32],
    w_ternary: &[f32],
    gamma: &[f32],
    beta: f32,
    bias: &[f32],
    batch: usize,
    d_in: usize,
    d_out: usize,
) -> Vec<f32> {
    bitlinear_forward_cpu(x_int8, w_ternary, gamma, beta, bias, batch, d_in, d_out)
}
