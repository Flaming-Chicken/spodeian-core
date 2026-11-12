//! CPU and WGSL kernels for Offline RL Imitation Cql IQL Gail and Decision Transformer.

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

/// IQL asymmetric expectile loss elementwise CPU kernel:
/// out[i] = |tau - I(diff[i] < 0)| * diff[i]^2.
pub fn iql_expectile_loss_cpu(diff: &[f32], tau: f32) -> Vec<f32> {
    let n = diff.len();
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let u = diff[i];
        let weight = if u > 0.0f32 { tau } else { 1.0f32 - tau };
        out[i] = weight * u * u;
    }
    out
}

/// IQL asymmetric expectile loss WGSL compute shader
pub const IQL_EXPECTILE_LOSS_WGSL: &str = r#"
struct Config {
    n: u32,
    tau: f32,
};

@group(0) @binding(0) var<storage, read> diff: array<f32>;
@group(0) @binding(1) var<storage, read_write> losses: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Config;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= cfg.n) {
        return;
    }

    let u = diff[idx];
    var weight: f32 = 1.0 - cfg.tau;
    if (u > 0.0) {
        weight = cfg.tau;
    }

    losses[idx] = weight * u * u;
}
"#;
