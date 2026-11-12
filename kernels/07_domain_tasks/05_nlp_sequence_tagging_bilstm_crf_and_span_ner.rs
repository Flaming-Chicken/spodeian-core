//! CPU (and optional WGSL) kernels for Nlp Sequence Tagging Bilstm Crf and Span Ner.

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

/// CPU span endpoint boundary feature extraction:
/// out[s] = [H[start_s]; H[end_s]; H[end_s] - H[start_s]]
pub fn span_features_cpu(
    seq_embeds: &[f32],
    start_idx: &[usize],
    end_idx: &[usize],
    num_spans: usize,
    d: usize,
    _seq_len: usize,
) -> Vec<f32> {
    let out_dim = 3 * d;
    let mut out = vec![0.0f32; num_spans * out_dim];

    for s in 0..num_spans {
        let s_idx = start_idx[s];
        let e_idx = end_idx[s];
        let row_offset = s * out_dim;

        let start_offset = s_idx * d;
        let end_offset = e_idx * d;

        for di in 0..d {
            let h_s = seq_embeds[start_offset + di];
            let h_e = seq_embeds[end_offset + di];
            out[row_offset + di] = h_s;
            out[row_offset + d + di] = h_e;
            out[row_offset + 2 * d + di] = h_e - h_s;
        }
    }
    out
}

/// WGSL compute shader for parallel span feature assembly.
pub const SPAN_FEATURES_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> seq_embeds_buf: array<f32>;
@group(0) @binding(1) var<storage, read> start_idx_buf: array<u32>;
@group(0) @binding(2) var<storage, read> end_idx_buf: array<u32>;
@group(0) @binding(3) var<storage, read_write> span_out_buf: array<f32>;

struct SpanParams {
    num_spans: u32,
    d: u32,
};
@group(0) @binding(4) var<uniform> params: SpanParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let s = global_id.x;
    if (s >= params.num_spans) {
        return;
    }

    let s_idx = start_idx_buf[s];
    let e_idx = end_idx_buf[s];
    let row_offset = s * (3u * params.d);

    let start_offset = s_idx * params.d;
    let end_offset = e_idx * params.d;

    for (var di: u32 = 0u; di < params.d; di = di + 1u) {
        let h_s = seq_embeds_buf[start_offset + di];
        let h_e = seq_embeds_buf[end_offset + di];
        span_out_buf[row_offset + di] = h_s;
        span_out_buf[row_offset + params.d + di] = h_e;
        span_out_buf[row_offset + 2u * params.d + di] = h_e - h_s;
    }
}
"#;
