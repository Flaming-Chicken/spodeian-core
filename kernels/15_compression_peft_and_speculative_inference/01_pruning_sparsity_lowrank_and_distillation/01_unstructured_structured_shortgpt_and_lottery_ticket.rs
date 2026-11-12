//! CPU and WGSL kernels for Unstructured Structured Shortgpt and Lottery Ticket.

/// WGSL compute shader for parallel magnitude thresholding and masking.
pub const PRUNING_MASK_WGSL: &str = r#"
struct Uniforms {
    total_elements: u32,
    threshold: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> w_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> w_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> mask_out: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn prune_mask_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= u.total_elements) { return; }

    let val = w_in[idx];
    if (abs(val) >= u.threshold) {
        w_out[idx] = val;
        mask_out[idx] = 1.0;
    } else {
        w_out[idx] = 0.0;
        mask_out[idx] = 0.0;
    }
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

/// CPU unstructured magnitude pruning.
pub fn unstructured_magnitude_prune_cpu(w: &[f32], sparsity: f32) -> (Vec<f32>, Vec<f32>) {
    let n = w.len();
    let k = ((sparsity * n as f32) as usize).max(1).min(n);
    let mut abs_vals: Vec<f32> = w.iter().map(|&x| x.abs()).collect();
    abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let tau = abs_vals[k - 1];

    let mut w_pruned = vec![0.0f32; n];
    let mut mask = vec![0.0f32; n];
    for i in 0..n {
        if w[i].abs() >= tau {
            w_pruned[i] = w[i];
            mask[i] = 1.0;
        }
    }
    (w_pruned, mask)
}

/// CPU cosine similarity and Block Influence (BI): 1 - mean(cos(x_in, x_out)).
pub fn compute_block_influence_cpu(x_in: &[f32], x_out: &[f32], batch: usize, dim: usize) -> f32 {
    let mut sum_cos = 0.0f32;
    for b in 0..batch {
        let in_row = &x_in[b * dim..(b + 1) * dim];
        let out_row = &x_out[b * dim..(b + 1) * dim];
        let mut dot = 0.0f32;
        let mut norm_in_sq = 0.0f32;
        let mut norm_out_sq = 0.0f32;
        for i in 0..dim {
            dot += in_row[i] * out_row[i];
            norm_in_sq += in_row[i] * in_row[i];
            norm_out_sq += out_row[i] * out_row[i];
        }
        let denom = (norm_in_sq.sqrt() * norm_out_sq.sqrt()) + 1e-8;
        sum_cos += dot / denom;
    }
    1.0 - (sum_cos / batch as f32)
}
