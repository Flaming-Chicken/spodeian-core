//! CPU and WGSL kernels for Binary Sub1bit and Ternary BNN XNOR BiLLM OneBit TWN TTQ.

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

/// WGSL compute shader for fused OneBit rank-1 sign-magnitude dequantization:
/// `W_deq[r, c] = signs[r, c] * (g[r] * h[c])`
pub const ONEBIT_DEQUANT_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> signs: array<f32>;
@group(0) @binding(1) var<storage, read> g_rows: array<f32>;
@group(0) @binding(2) var<storage, read> h_cols: array<f32>;
@group(0) @binding(3) var<storage, read_write> w_deq: array<f32>;

struct Uniforms {
    d_in: u32,
    d_out: u32,
};
@group(0) @binding(4) var<uniform> params: Uniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let r = global_id.x;
    let c = global_id.y;
    if (r >= params.d_in || c >= params.d_out) {
        return;
    }
    let idx = r * params.d_out + c;
    w_deq[idx] = signs[idx] * (g_rows[r] * h_cols[c]);
}
"#;

/// CPU reference implementation of fused OneBit dequantization.
pub fn onebit_dequant_cpu(
    signs: &[f32],
    g_rows: &[f32],
    h_cols: &[f32],
    d_in: usize,
    d_out: usize,
) -> Vec<f32> {
    let mut w_deq = vec![0.0f32; d_in * d_out];
    for r in 0..d_in {
        for c in 0..d_out {
            let idx = r * d_out + c;
            w_deq[idx] = signs[idx] * (g_rows[r] * h_cols[c]);
        }
    }
    w_deq
}

/// Auto dispatcher: returns CPU reference fallback.
pub fn onebit_dequant_auto(
    signs: &[f32],
    g_rows: &[f32],
    h_cols: &[f32],
    d_in: usize,
    d_out: usize,
) -> Vec<f32> {
    onebit_dequant_cpu(signs, g_rows, h_cols, d_in, d_out)
}
