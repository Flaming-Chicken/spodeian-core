//! CPU and WGSL kernels for RLHF Reward Modeling and PPO vs Rloo.

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

/// RLOO Leave-One-Out advantage CPU kernel:
/// adv[b, k] = scores[b, k] - (sum_j scores[b, j] - scores[b, k]) / (K - 1).
pub fn rloo_leave_one_out_advantages_cpu(scores: &[f32], batch: usize, k: usize) -> Vec<f32> {
    let mut adv = vec![0.0f32; batch * k];
    let denom = (k - 1) as f32;

    for bi in 0..batch {
        let offset = bi * k;
        let mut sum_score = 0.0f32;
        for ki in 0..k {
            sum_score += scores[offset + ki];
        }
        for ki in 0..k {
            let baseline = (sum_score - scores[offset + ki]) / denom;
            adv[offset + ki] = scores[offset + ki] - baseline;
        }
    }
    adv
}

/// RLOO Leave-One-Out advantage WGSL compute shader
pub const RLOO_ADVANTAGES_WGSL: &str = r#"
struct Config {
    batch: u32,
    k: u32,
};

@group(0) @binding(0) var<storage, read> scores: array<f32>;
@group(0) @binding(1) var<storage, read_write> advantages: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Config;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let bi = global_id.x;
    if (bi >= cfg.batch) {
        return;
    }

    let offset = bi * cfg.k;
    var sum_score: f32 = 0.0;
    for (var ki: u32 = 0u; ki < cfg.k; ki = ki + 1u) {
        sum_score = sum_score + scores[offset + ki];
    }

    let denom = f32(cfg.k - 1u);
    for (var ki: u32 = 0u; ki < cfg.k; ki = ki + 1u) {
        let s = scores[offset + ki];
        let baseline = (sum_score - s) / denom;
        advantages[offset + ki] = s - baseline;
    }
}
"#;
