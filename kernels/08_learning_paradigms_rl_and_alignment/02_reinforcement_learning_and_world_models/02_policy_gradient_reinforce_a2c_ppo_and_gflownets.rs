//! CPU and WGSL kernels for Policy Gradient Reinforce A2c PPO and GFlowNets.

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

/// PPO clipped surrogate objective elementwise CPU kernel:
/// out[i] = min(r_i * A_i, clamp(r_i, 1-eps, 1+eps) * A_i) where r_i = exp(log_p[i] - old_log_p[i]).
pub fn ppo_clip_surrogate_cpu(
    log_probs: &[f32],
    old_log_probs: &[f32],
    advantages: &[f32],
    clip_eps: f32,
) -> Vec<f32> {
    let n = log_probs.len();
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let ratio = (log_probs[i] - old_log_probs[i]).exp();
        let surr1 = ratio * advantages[i];
        let clamped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps);
        let surr2 = clamped * advantages[i];
        out[i] = surr1.min(surr2);
    }
    out
}

/// PPO clipped surrogate objective WGSL compute shader
pub const PPO_CLIP_OBJECTIVE_WGSL: &str = r#"
struct Config {
    n: u32,
    clip_eps: f32,
};

@group(0) @binding(0) var<storage, read> log_probs: array<f32>;
@group(0) @binding(1) var<storage, read> old_log_probs: array<f32>;
@group(0) @binding(2) var<storage, read> advantages: array<f32>;
@group(0) @binding(3) var<storage, read_write> surrogates: array<f32>;
@group(0) @binding(4) var<uniform> cfg: Config;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= cfg.n) {
        return;
    }

    let ratio = exp(log_probs[idx] - old_log_probs[idx]);
    let surr1 = ratio * advantages[idx];
    let clamped_ratio = clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps);
    let surr2 = clamped_ratio * advantages[idx];

    surrogates[idx] = min(surr1, surr2);
}
"#;
