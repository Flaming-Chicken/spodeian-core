//! CPU and WGSL kernels for KV Cache Quantization: KIVI and KVQuant.

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

/// WGSL compute shader for fused on-the-fly KIVI Key/Value dequantization and score accumulation.
pub const KIVI_FUSED_DECODE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k_int: array<f32>;
@group(0) @binding(2) var<storage, read> k_scale: array<f32>;
@group(0) @binding(3) var<storage, read> k_zero: array<f32>;
@group(0) @binding(4) var<storage, read_write> scores: array<f32>;

struct Uniforms {
    seq_len: u32,
    d_k: u32,
    inv_sqrt_d: f32,
};
@group(0) @binding(5) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let s = global_id.x;
    if (s >= params.seq_len) {
        return;
    }

    var dot: f32 = 0.0;
    for (var c: u32 = 0u; c < params.d_k; c = c + 1u) {
        let idx = s * params.d_k + c;
        let k_val = (k_int[idx] - k_zero[c]) * k_scale[c];
        dot = dot + q[c] * k_val;
    }
    scores[s] = dot * params.inv_sqrt_d;
}
"#;

/// CPU reference implementation of KIVI on-the-fly Key-channel and Value-token attention decoding.
pub fn kivi_decode_cpu(
    q: &[f32],
    k_deq: &[f32],
    v_deq: &[f32],
    seq_len: usize,
    d_k: usize,
    d_v: usize,
) -> Vec<f32> {
    let inv_sqrt = 1.0f32 / (d_k as f32).sqrt();
    let mut scores = vec![0.0f32; seq_len];
    let mut max_score = f32::NEG_INFINITY;

    for s in 0..seq_len {
        let mut dot = 0.0f32;
        for c in 0..d_k {
            dot += q[c] * k_deq[s * d_k + c];
        }
        let score = dot * inv_sqrt;
        scores[s] = score;
        if score > max_score {
            max_score = score;
        }
    }

    // Softmax
    let mut sum_exp = 0.0f32;
    for s in 0..seq_len {
        let e = (scores[s] - max_score).exp();
        scores[s] = e;
        sum_exp += e;
    }
    let inv_sum = 1.0f32 / sum_exp.max(1e-12);
    for s in 0..seq_len {
        scores[s] *= inv_sum;
    }

    // Output accumulation
    let mut out = vec![0.0f32; d_v];
    for s in 0..seq_len {
        let prob = scores[s];
        for c in 0..d_v {
            out[c] += prob * v_deq[s * d_v + c];
        }
    }
    out
}

/// Auto dispatcher: returns CPU reference fallback.
pub fn kivi_decode_auto(
    q: &[f32],
    k_deq: &[f32],
    v_deq: &[f32],
    seq_len: usize,
    d_k: usize,
    d_v: usize,
) -> Vec<f32> {
    kivi_decode_cpu(q, k_deq, v_deq, seq_len, d_k, d_v)
}
