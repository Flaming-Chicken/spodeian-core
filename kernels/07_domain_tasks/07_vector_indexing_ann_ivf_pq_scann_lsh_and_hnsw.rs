//! CPU (and optional WGSL) kernels for Vector Indexing Ann IVF PQ Scann Lsh and HNSW.

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

/// Fused Product Quantization Asymmetric Distance Computation (ADC) on CPU:
/// out[i] = sum_m dist_table[m, codes[i, m]]
pub fn pq_adc_cpu(
    dist_table: &[f32],
    codes: &[u32],
    num_docs: usize,
    m: usize,
    k: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; num_docs];
    for i in 0..num_docs {
        let mut sum_dist = 0.0f32;
        let doc_offset = i * m;
        for mi in 0..m {
            let cluster_id = codes[doc_offset + mi] as usize;
            sum_dist += dist_table[mi * k + cluster_id];
        }
        out[i] = sum_dist;
    }
    out
}

/// WGSL compute shader for parallel Product Quantization Asymmetric Distance Computation.
pub const PQ_ADC_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> dist_table_buf: array<f32>;
@group(0) @binding(1) var<storage, read> codes_buf: array<u32>;
@group(0) @binding(2) var<storage, read_write> out_dist_buf: array<f32>;

struct PqParams {
    num_docs: u32,
    m: u32,
    k: u32,
};
@group(0) @binding(3) var<uniform> params: PqParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.num_docs) {
        return;
    }

    var sum_dist: f32 = 0.0;
    let doc_offset = i * params.m;

    for (var mi: u32 = 0u; mi < params.m; mi = mi + 1u) {
        let cluster_id = codes_buf[doc_offset + mi];
        let table_idx = mi * params.k + cluster_id;
        sum_dist = sum_dist + dist_table_buf[table_idx];
    }

    out_dist_buf[i] = sum_dist;
}
"#;
