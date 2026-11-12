//! CPU and WGSL kernels for Multi Agent RL QMIX Mappo Cfr and Agentic Reasoning Loops.

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

/// CFR Regret Matching CPU kernel.
pub fn cfr_regret_matching_cpu(regrets: &[f32], b: usize, a_dim: usize, eps: f32) -> Vec<f32> {
    let mut strat = vec![0.0f32; b * a_dim];
    for bi in 0..b {
        let offset = bi * a_dim;
        let mut sum_pos = 0.0f32;
        for ai in 0..a_dim {
            let r = regrets[offset + ai];
            if r > 0.0 {
                sum_pos += r;
            }
        }

        if sum_pos > eps {
            for ai in 0..a_dim {
                let r = regrets[offset + ai].max(0.0f32);
                strat[offset + ai] = r / (sum_pos + eps);
            }
        } else {
            let uniform = 1.0f32 / (a_dim as f32);
            for ai in 0..a_dim {
                strat[offset + ai] = uniform;
            }
        }
    }
    strat
}

/// CFR Regret Matching WGSL compute shader
pub const CFR_REGRET_MATCHING_WGSL: &str = r#"
struct Config {
    b: u32,
    a_dim: u32,
    eps: f32,
};

@group(0) @binding(0) var<storage, read> regrets: array<f32>;
@group(0) @binding(1) var<storage, read_write> strategy: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Config;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let bi = global_id.x;
    if (bi >= cfg.b) {
        return;
    }

    let offset = bi * cfg.a_dim;
    var sum_pos: f32 = 0.0;
    for (var a: u32 = 0u; a < cfg.a_dim; a = a + 1u) {
        let r = regrets[offset + a];
        if (r > 0.0) {
            sum_pos = sum_pos + r;
        }
    }

    if (sum_pos > cfg.eps) {
        for (var a: u32 = 0u; a < cfg.a_dim; a = a + 1u) {
            let r = max(regrets[offset + a], 0.0);
            strategy[offset + a] = r / (sum_pos + cfg.eps);
        }
    } else {
        let uniform = 1.0 / f32(cfg.a_dim);
        for (var a: u32 = 0u; a < cfg.a_dim; a = a + 1u) {
            strategy[offset + a] = uniform;
        }
    }
}
"#;
