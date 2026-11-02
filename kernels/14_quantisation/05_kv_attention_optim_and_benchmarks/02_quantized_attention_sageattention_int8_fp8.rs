//! CPU and WGSL kernels for Quantized Attention: SageAttention INT8/FP8.

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

/// WGSL compute shader for fused SageAttention INT8 QK^T score computation.
pub const SAGEATTN_SCORE_GEMM_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> q_int8: array<f32>;
@group(0) @binding(1) var<storage, read> k_int8: array<f32>;
@group(0) @binding(2) var<storage, read_write> scores: array<f32>;

struct Uniforms {
    batch: u32,
    seq_len: u32,
    d_k: u32,
    scale: f32,
};
@group(0) @binding(3) var<uniform> params: Uniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.x;
    let s = global_id.y;
    if (b >= params.batch || s >= params.seq_len) {
        return;
    }

    var acc: f32 = 0.0;
    for (var c: u32 = 0u; c < params.d_k; c = c + 1u) {
        let q_val = q_int8[b * params.d_k + c];
        let k_val = k_int8[s * params.d_k + c];
        acc = acc + q_val * k_val;
    }

    let idx = b * params.seq_len + s;
    scores[idx] = acc * params.scale;
}
"#;

/// CPU reference implementation of SageAttention INT8 QK^T -> Softmax -> PV forward pass.
pub fn sage_attention_cpu(
    q_int8: &[f32],
    k_int8: &[f32],
    v_deq: &[f32],
    batch: usize,
    seq_len: usize,
    d_k: usize,
    d_v: usize,
    scale: f32,
) -> Vec<f32> {
    let mut scores = vec![0.0f32; batch * seq_len];

    for b in 0..batch {
        for s in 0..seq_len {
            let mut acc = 0.0f32;
            for c in 0..d_k {
                acc += q_int8[b * d_k + c] * k_int8[s * d_k + c];
            }
            scores[b * seq_len + s] = acc * scale;
        }
    }

    // Softmax per row
    let mut probs = vec![0.0f32; batch * seq_len];
    for b in 0..batch {
        let row_start = b * seq_len;
        let mut max_val = f32::NEG_INFINITY;
        for s in 0..seq_len {
            if scores[row_start + s] > max_val {
                max_val = scores[row_start + s];
            }
        }
        let mut sum_exp = 0.0f32;
        for s in 0..seq_len {
            let e = (scores[row_start + s] - max_val).exp();
            probs[row_start + s] = e;
            sum_exp += e;
        }
        let inv_sum = 1.0f32 / sum_exp.max(1e-12);
        for s in 0..seq_len {
            probs[row_start + s] *= inv_sum;
        }
    }

    // Out = probs @ v_deq: [batch, d_v]
    let mut out = vec![0.0f32; batch * d_v];
    for b in 0..batch {
        for o in 0..d_v {
            let mut acc = 0.0f32;
            for s in 0..seq_len {
                acc += probs[b * seq_len + s] * v_deq[s * d_v + o];
            }
            out[b * d_v + o] = acc;
        }
    }
    out
}

/// Auto dispatcher: returns CPU reference fallback.
pub fn sage_attention_auto(
    q_int8: &[f32],
    k_int8: &[f32],
    v_deq: &[f32],
    batch: usize,
    seq_len: usize,
    d_k: usize,
    d_v: usize,
    scale: f32,
) -> Vec<f32> {
    sage_attention_cpu(q_int8, k_int8, v_deq, batch, seq_len, d_k, d_v, scale)
}
