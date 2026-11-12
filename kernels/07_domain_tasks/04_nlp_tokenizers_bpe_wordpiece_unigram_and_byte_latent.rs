//! CPU (and optional WGSL) kernels for Nlp Tokenizers Bpe Wordpiece Unigram and Byte Latent.

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

/// Fused Shannon entropy computation across vocabulary for boundary detection:
/// H = -sum_v p_v * log2(p_v + eps)
pub fn entropy_boundary_detector_cpu(
    probs: &[f32],
    total_positions: usize,
    vocab_size: usize,
    eps: f32,
) -> Vec<f32> {
    let mut entropy = vec![0.0f32; total_positions];
    for p in 0..total_positions {
        let mut h = 0.0f32;
        let offset = p * vocab_size;
        for v in 0..vocab_size {
            let prob = probs[offset + v];
            h -= prob * (prob + eps).log2();
        }
        entropy[p] = h;
    }
    entropy
}

/// WGSL compute shader for parallel local byte entropy evaluation across sequence tokens.
pub const ENTROPY_BOUNDARY_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> probs_buf: array<f32>;
@group(0) @binding(1) var<storage, read_write> entropy_buf: array<f32>;

struct EntropyParams {
    total_positions: u32,
    vocab_size: u32,
    eps: f32,
};
@group(0) @binding(2) var<uniform> params: EntropyParams;

const INV_LN2: f32 = 1.4426950408889634; // 1.0 / ln(2)

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let p = global_id.x;
    if (p >= params.total_positions) {
        return;
    }

    let offset = p * params.vocab_size;
    var h: f32 = 0.0;
    for (var v: u32 = 0u; v < params.vocab_size; v = v + 1u) {
        let prob = probs_buf[offset + v];
        let log2_p = log(prob + params.eps) * INV_LN2;
        h = h - prob * log2_p;
    }
    entropy_buf[p] = h;
}
"#;
