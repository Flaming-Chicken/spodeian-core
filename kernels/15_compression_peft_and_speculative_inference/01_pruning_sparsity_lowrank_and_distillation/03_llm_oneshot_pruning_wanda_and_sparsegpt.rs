//! CPU and WGSL kernels for LLM Oneshot Pruning Wanda and Sparsegpt.

/// WGSL compute shader for parallel Wanda importance scoring and threshold masking.
pub const WANDA_SCORING_WGSL: &str = r#"
struct Uniforms {
    d_out: u32,
    d_in: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> w_in: array<f32>; // [d_out, d_in]
@group(0) @binding(2) var<storage, read> act_norms: array<f32>; // [d_in]
@group(0) @binding(3) var<storage, read_write> scores_out: array<f32>; // [d_out, d_in]

@compute @workgroup_size(16, 16, 1)
fn wanda_score_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    let c = gid.y;
    if (r >= u.d_out || c >= u.d_in) { return; }

    let idx = r * u.d_in + c;
    let w_val = abs(w_in[idx]);
    let norm_val = act_norms[c];
    scores_out[idx] = w_val * norm_val;
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

/// CPU activation column norm computation.
pub fn compute_activation_norms_cpu(x: &[f32], n_samples: usize, d_in: usize) -> Vec<f32> {
    let mut norms = vec![0.0f32; d_in];
    for c in 0..d_in {
        let mut sum_sq = 0.0f32;
        for r in 0..n_samples {
            let val = x[r * d_in + c];
            sum_sq += val * val;
        }
        norms[c] = sum_sq.sqrt();
    }
    norms
}

/// CPU Wanda row-wise one-shot pruning.
pub fn wanda_prune_cpu(
    w: &[f32],
    act_norms: &[f32],
    d_out: usize,
    d_in: usize,
    sparsity: f32,
) -> (Vec<f32>, Vec<f32>) {
    let k = ((sparsity * d_in as f32) as usize).max(1);
    let mut w_pruned = vec![0.0f32; d_out * d_in];
    let mut mask = vec![0.0f32; d_out * d_in];

    for r in 0..d_out {
        let mut row_scores: Vec<(usize, f32)> = (0..d_in)
            .map(|c| (c, w[r * d_in + c].abs() * act_norms[c]))
            .collect();
        row_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let tau = row_scores[k - 1].1;
        for c in 0..d_in {
            let idx = r * d_in + c;
            let s = w[idx].abs() * act_norms[c];
            if s >= tau {
                w_pruned[idx] = w[idx];
                mask[idx] = 1.0;
            }
        }
    }

    (w_pruned, mask)
}
