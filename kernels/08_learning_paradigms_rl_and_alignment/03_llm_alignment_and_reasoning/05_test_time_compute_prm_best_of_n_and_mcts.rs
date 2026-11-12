//! CPU and WGSL kernels for Test Time Compute PRM Best of N and MCTS.

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

/// Process Reward Model product aggregation on CPU.
pub fn prm_product_aggregation_cpu(
    step_probs: &[f32],
    step_mask: &[f32],
    batch: usize,
    steps: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(batch);
    for b in 0..batch {
        let mut sum_log = 0.0f32;
        for s in 0..steps {
            let idx = b * steps + s;
            let p = step_probs[idx].max(eps);
            sum_log += step_mask[idx] * p.ln();
        }
        out.push(sum_log.exp());
    }
    out
}

/// Process Reward Model min aggregation on CPU.
pub fn prm_min_aggregation_cpu(
    step_probs: &[f32],
    step_mask: &[f32],
    batch: usize,
    steps: usize,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(batch);
    for b in 0..batch {
        let mut min_val = 1.0f32;
        for s in 0..steps {
            let idx = b * steps + s;
            if step_mask[idx] > 0.5 {
                min_val = min_val.min(step_probs[idx]);
            }
        }
        out.push(min_val);
    }
    out
}

/// WGSL shader for parallel PUCT node score computation.
pub const PUCT_SELECTION_WGSL: &str = r#"
struct Params {
    num_actions: u32,
    c_puct: f32,
    sqrt_total_visits: f32,
    pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> q_values: array<f32>;
@group(0) @binding(2) var<storage, read> priors: array<f32>;
@group(0) @binding(3) var<storage, read> visit_counts: array<f32>;
@group(0) @binding(4) var<storage, read_write> puct_scores: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let a = gid.x;
    if (a >= params.num_actions) {
        return;
    }
    let u = q_values[a] + params.c_puct * priors[a] * params.sqrt_total_visits / (1.0 + visit_counts[a]);
    puct_scores[a] = u;
}
"#;

/// CPU reference PUCT node selection.
pub fn puct_selection_cpu(
    q_values: &[f32],
    priors: &[f32],
    visit_counts: &[f32],
    c_puct: f32,
) -> (Vec<f32>, usize) {
    let a = q_values.len();
    let total_visits: f32 = visit_counts.iter().sum();
    let sqrt_total = total_visits.sqrt();

    let mut scores = Vec::with_capacity(a);
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;

    for i in 0..a {
        let u = q_values[i] + c_puct * priors[i] * sqrt_total / (1.0 + visit_counts[i]);
        scores.push(u);
        if u > best_val {
            best_val = u;
            best_idx = i;
        }
    }
    (scores, best_idx)
}

/// Automatic dispatcher falling back to CPU.
pub fn puct_selection_auto(
    q_values: &[f32],
    priors: &[f32],
    visit_counts: &[f32],
    c_puct: f32,
) -> (Vec<f32>, usize) {
    puct_selection_cpu(q_values, priors, visit_counts, c_puct)
}
