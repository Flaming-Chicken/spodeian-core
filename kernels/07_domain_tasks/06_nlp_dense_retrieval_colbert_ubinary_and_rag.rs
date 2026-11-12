//! CPU (and optional WGSL) kernels for Nlp Dense Retrieval Colbert Ubinary and RAG.

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

/// Fused ColBERT MaxSim Late Interaction on CPU:
/// Score = sum_i max_j (q_i . d_j)
pub fn colbert_maxsim_cpu(
    query_embeds: &[f32],
    doc_embeds: &[f32],
    q_len: usize,
    doc_len: usize,
    d: usize,
) -> f32 {
    let mut total_score = 0.0f32;

    for qi in 0..q_len {
        let q_offset = qi * d;
        let mut max_sim = f32::NEG_INFINITY;

        for dj in 0..doc_len {
            let d_offset = dj * d;
            let mut dot = 0.0f32;
            for k in 0..d {
                dot += query_embeds[q_offset + k] * doc_embeds[d_offset + k];
            }
            if dot > max_sim {
                max_sim = dot;
            }
        }
        total_score += max_sim;
    }

    total_score
}

/// WGSL compute shader for parallel ColBERT MaxSim per-query token maximum similarity evaluation.
pub const COLBERT_MAXSIM_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> query_buf: array<f32>;
@group(0) @binding(1) var<storage, read> doc_buf: array<f32>;
@group(0) @binding(2) var<storage, read_write> max_sims_buf: array<f32>;

struct MaxSimParams {
    q_len: u32,
    doc_len: u32,
    d: u32,
};
@group(0) @binding(3) var<uniform> params: MaxSimParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let qi = global_id.x;
    if (qi >= params.q_len) {
        return;
    }

    let q_offset = qi * params.d;
    var max_sim: f32 = -1e9;

    for (var dj: u32 = 0u; dj < params.doc_len; dj = dj + 1u) {
        let d_offset = dj * params.d;
        var dot: f32 = 0.0;
        for (var k: u32 = 0u; k < params.d; k = k + 1u) {
            dot = dot + query_buf[q_offset + k] * doc_buf[d_offset + k];
        }
        max_sim = max(max_sim, dot);
    }

    max_sims_buf[qi] = max_sim;
}
"#;
