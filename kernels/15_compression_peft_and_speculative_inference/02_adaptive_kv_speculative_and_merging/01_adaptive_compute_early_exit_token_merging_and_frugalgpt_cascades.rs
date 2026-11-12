//! CPU and WGSL kernels for Adaptive Compute Early Exit Token Merging and FrugalGPT Cascades.

/// WGSL compute shader for bipartite token similarity matrix computation in Token Merging (ToMe).
pub const TOME_BIPARTITE_SIM_WGSL: &str = r#"
struct Uniforms {
    batch: u32,
    len_a: u32,
    len_b: u32,
    dim: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> a_tokens: array<f32>; // [batch, len_a, dim]
@group(0) @binding(2) var<storage, read> b_tokens: array<f32>; // [batch, len_b, dim]
@group(0) @binding(3) var<storage, read_write> sim_matrix: array<f32>; // [batch, len_a, len_b]

@compute @workgroup_size(16, 16, 1)
fn bipartite_sim_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; // index in A
    let j = gid.y; // index in B
    let b = gid.z; // batch index

    if (b >= u.batch || i >= u.len_a || j >= u.len_b) { return; }

    let offset_a = (b * u.len_a + i) * u.dim;
    let offset_b = (b * u.len_b + j) * u.dim;

    var dot: f32 = 0.0;
    var norm_a: f32 = 0.0;
    var norm_b: f32 = 0.0;

    for (var d = 0u; d < u.dim; d++) {
        let va = a_tokens[offset_a + d];
        let vb = b_tokens[offset_b + d];
        dot += va * vb;
        norm_a += va * va;
        norm_b += vb * vb;
    }

    norm_a = max(sqrt(norm_a), 1e-8);
    norm_b = max(sqrt(norm_b), 1e-8);

    let out_idx = (b * u.len_a + i) * u.len_b + j;
    sim_matrix[out_idx] = dot / (norm_a * norm_b);
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

/// CPU implementation of bipartite cosine similarity computation for ToMe.
pub fn tome_bipartite_sim_cpu(
    a_tokens: &[f32],
    b_tokens: &[f32],
    batch: usize,
    len_a: usize,
    len_b: usize,
    dim: usize,
) -> Vec<f32> {
    let mut sim = vec![0.0f32; batch * len_a * len_b];
    for b in 0..batch {
        let b_offset_a = b * len_a * dim;
        let b_offset_b = b * len_b * dim;
        for i in 0..len_a {
            let a_slice = &a_tokens[b_offset_a + i * dim..b_offset_a + (i + 1) * dim];
            let norm_a = a_slice.iter().map(|&v| v * v).sum::<f32>().sqrt().max(1e-8);
            for j in 0..len_b {
                let b_slice = &b_tokens[b_offset_b + j * dim..b_offset_b + (j + 1) * dim];
                let norm_b = b_slice.iter().map(|&v| v * v).sum::<f32>().sqrt().max(1e-8);
                let dot: f32 = a_slice
                    .iter()
                    .zip(b_slice.iter())
                    .map(|(&x, &y)| x * y)
                    .sum();
                sim[(b * len_a + i) * len_b + j] = dot / (norm_a * norm_b);
            }
        }
    }
    sim
}

/// CPU early exit decision evaluator.
pub fn early_exit_cpu(
    probs: &[f32],
    batch: usize,
    classes: usize,
    threshold: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut confidences = vec![0.0f32; batch];
    let mut exits = vec![0.0f32; batch];
    for b in 0..batch {
        let row = &probs[b * classes..(b + 1) * classes];
        let max_val = row.iter().cloned().fold(0.0f32, f32::max);
        confidences[b] = max_val;
        exits[b] = if max_val >= threshold { 1.0 } else { 0.0 };
    }
    (confidences, exits)
}
