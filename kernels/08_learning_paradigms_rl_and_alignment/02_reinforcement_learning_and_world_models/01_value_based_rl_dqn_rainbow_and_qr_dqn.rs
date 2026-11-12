//! CPU and WGSL kernels for Value Based RL Dqn Rainbow and Qr Dqn.

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

/// Dueling Q-value aggregation CPU kernel:
/// Q(s, a) = V(s) + A(s, a) - (1 / A) * sum_{a'} A(s, a')
/// value: [B] (or [B, 1])
/// advantage: [B, A]
/// out: [B, A]
pub fn dueling_q_values_cpu(value: &[f32], advantage: &[f32], b: usize, a: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; b * a];
    let inv_a = 1.0f32 / (a as f32);

    for bi in 0..b {
        let v = value[bi];
        let offset = bi * a;
        let mut sum_adv = 0.0f32;
        for ai in 0..a {
            sum_adv += advantage[offset + ai];
        }
        let mean_adv = sum_adv * inv_a;
        for ai in 0..a {
            out[offset + ai] = v + advantage[offset + ai] - mean_adv;
        }
    }
    out
}

/// Dueling Q-value aggregation WGSL compute shader
pub const DUELING_Q_WGSL: &str = r#"
struct Config {
    b: u32,
    a: u32,
};

@group(0) @binding(0) var<storage, read> value: array<f32>;
@group(0) @binding(1) var<storage, read> advantage: array<f32>;
@group(0) @binding(2) var<storage, read_write> q_values: array<f32>;
@group(0) @binding(3) var<uniform> cfg: Config;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ai = global_id.x;
    let bi = global_id.y;

    if (bi >= cfg.b || ai >= cfg.a) {
        return;
    }

    let offset = bi * cfg.a;
    var sum_adv: f32 = 0.0;
    for (var i: u32 = 0u; i < cfg.a; i = i + 1u) {
        sum_adv = sum_adv + advantage[offset + i];
    }
    let mean_adv = sum_adv / f32(cfg.a);

    let v = value[bi];
    let adv = advantage[offset + ai];
    q_values[offset + ai] = v + adv - mean_adv;
}
"#;
