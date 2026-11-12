//! CPU and WGSL kernels for GRPO Dapo and Verifiable Reward Reasoning.

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

/// GRPO Group Advantages CPU kernel:
/// adv[b, g] = (rewards[b, g] - mean) / sqrt(variance + eps).
pub fn grpo_group_advantages_cpu(rewards: &[f32], batch: usize, g: usize, eps: f32) -> Vec<f32> {
    let mut adv = vec![0.0f32; batch * g];
    for bi in 0..batch {
        let offset = bi * g;
        let mut sum_r = 0.0f32;
        for gi in 0..g {
            sum_r += rewards[offset + gi];
        }
        let mean = sum_r / (g as f32);

        let mut var_sum = 0.0f32;
        for gi in 0..g {
            let diff = rewards[offset + gi] - mean;
            var_sum += diff * diff;
        }
        let std = (var_sum / (g as f32) + eps).sqrt();

        for gi in 0..g {
            adv[offset + gi] = (rewards[offset + gi] - mean) / std;
        }
    }
    adv
}

/// GRPO Group Advantages WGSL compute shader
pub const GRPO_GROUP_ADVANTAGES_WGSL: &str = r#"
struct Config {
    batch: u32,
    g: u32,
    eps: f32,
};

@group(0) @binding(0) var<storage, read> rewards: array<f32>;
@group(0) @binding(1) var<storage, read_write> advantages: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Config;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let bi = global_id.x;
    if (bi >= cfg.batch) {
        return;
    }

    let offset = bi * cfg.g;
    var sum_r: f32 = 0.0;
    for (var gi: u32 = 0u; gi < cfg.g; gi = gi + 1u) {
        sum_r = sum_r + rewards[offset + gi];
    }
    let mean = sum_r / f32(cfg.g);

    var var_sum: f32 = 0.0;
    for (var gi: u32 = 0u; gi < cfg.g; gi = gi + 1u) {
        let diff = rewards[offset + gi] - mean;
        var_sum = var_sum + diff * diff;
    }
    let std = sqrt(var_sum / f32(cfg.g) + cfg.eps);

    for (var gi: u32 = 0u; gi < cfg.g; gi = gi + 1u) {
        advantages[offset + gi] = (rewards[offset + gi] - mean) / std;
    }
}
"#;
