//! CPU and WGSL kernels for Hierarchical RL Options Feudal Her and POMDP Active Inference.

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

/// FeUdal directional reward CPU kernel:
/// out[b] = (delta_s[b] . g[b]) / max(||delta_s[b]|| * ||g[b]||, eps).
pub fn feudal_directional_reward_cpu(
    delta_s: &[f32],
    subgoal_g: &[f32],
    b: usize,
    d: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; b];
    for bi in 0..b {
        let offset = bi * d;
        let mut dot = 0.0f32;
        let mut n_ds_sq = 0.0f32;
        let mut n_g_sq = 0.0f32;
        for di in 0..d {
            let ds = delta_s[offset + di];
            let g = subgoal_g[offset + di];
            dot += ds * g;
            n_ds_sq += ds * ds;
            n_g_sq += g * g;
        }
        let denom = (n_ds_sq.sqrt() * n_g_sq.sqrt()).max(eps);
        out[bi] = dot / denom;
    }
    out
}

/// FeUdal directional reward WGSL compute shader
pub const FEUDAL_DIRECTIONAL_REWARD_WGSL: &str = r#"
struct Config {
    b: u32,
    d: u32,
    eps: f32,
};

@group(0) @binding(0) var<storage, read> delta_s: array<f32>;
@group(0) @binding(1) var<storage, read> subgoal_g: array<f32>;
@group(0) @binding(2) var<storage, read_write> rewards: array<f32>;
@group(0) @binding(3) var<uniform> cfg: Config;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let bi = global_id.x;
    if (bi >= cfg.b) {
        return;
    }

    let offset = bi * cfg.d;
    var dot: f32 = 0.0;
    var n_ds_sq: f32 = 0.0;
    var n_g_sq: f32 = 0.0;

    for (var di: u32 = 0u; di < cfg.d; di = di + 1u) {
        let ds = delta_s[offset + di];
        let g = subgoal_g[offset + di];
        dot = dot + ds * g;
        n_ds_sq = n_ds_sq + ds * ds;
        n_g_sq = n_g_sq + g * g;
    }

    let denom = max(sqrt(n_ds_sq) * sqrt(n_g_sq), cfg.eps);
    rewards[bi] = dot / denom;
}
"#;
