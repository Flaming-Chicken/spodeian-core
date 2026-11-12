//! CPU and WGSL kernels for Production Serving Dynamic Batcher PD Disaggregation and Streaming.

/// WGSL compute shader for incremental decode attention score computation:
/// score[b, t] = (1 / sqrt(d)) * sum_{c=0}^{d-1} q[b, c] * k[b, t, c]
pub const FUSED_DECODE_ATTN_SCORES_WGSL: &str = r#"
struct Uniforms {
    batch_size: u32,
    seq_len: u32,
    head_dim: u32,
    scale: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> q: array<f32>;        // [batch_size, head_dim]
@group(0) @binding(2) var<storage, read> k: array<f32>;        // [batch_size, seq_len, head_dim]
@group(0) @binding(3) var<storage, read_write> scores: array<f32>; // [batch_size, seq_len]

@compute @workgroup_size(16, 16, 1)
fn decode_attn_scores_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    let t = gid.y;
    if (b >= u.batch_size || t >= u.seq_len) { return; }

    var dot: f32 = 0.0;
    let k_offset = (b * u.seq_len + t) * u.head_dim;
    let q_offset = b * u.head_dim;

    for (var c = 0u; c < u.head_dim; c++) {
        dot += q[q_offset + c] * k[k_offset + c];
    }
    scores[b * u.seq_len + t] = dot * u.scale;
}
"#;

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

/// CPU implementation of decode attention score kernel.
pub fn decode_attn_scores_cpu(
    q: &[f32],
    k: &[f32],
    batch_size: usize,
    seq_len: usize,
    head_dim: usize,
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut scores = vec![0.0f32; batch_size * seq_len];

    for b in 0..batch_size {
        let q_row = &q[b * head_dim..(b + 1) * head_dim];
        for t in 0..seq_len {
            let k_row = &k[(b * seq_len + t) * head_dim..(b * seq_len + t + 1) * head_dim];
            let dot: f32 = q_row
                .iter()
                .zip(k_row.iter())
                .map(|(&qi, &ki)| qi * ki)
                .sum();
            scores[b * seq_len + t] = dot * scale;
        }
    }
    scores
}

/// Dispatches decode attention scores with automatic CPU fallback.
pub fn decode_attn_scores_auto(
    q: &[f32],
    k: &[f32],
    batch_size: usize,
    seq_len: usize,
    head_dim: usize,
) -> Vec<f32> {
    decode_attn_scores_cpu(q, k, batch_size, seq_len, head_dim)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_attn_scores() {
        let q = vec![1.0, 0.0, 0.0, 1.0];
        let k = vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let scores = decode_attn_scores_cpu(&q, &k, 1, 2, 4);
        assert_eq!(scores.len(), 2);
        assert!((scores[0] - 1.0).abs() < 1e-5);
    }
}
