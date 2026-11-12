//! CPU and WGSL kernels for Successor Representations Forward Backward Zero Shot RL and Diayn.

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

/// Successor Features TD Error CPU kernel:
/// out[b, d] = phi[b, d] + gamma * (1 - done[b]) * psi_next[b, d] - psi_current[b, d].
pub fn successor_features_td_error_cpu(
    psi_current: &[f32],
    features: &[f32],
    psi_next: &[f32],
    dones: &[f32],
    batch: usize,
    d: usize,
    gamma: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * d];
    for bi in 0..batch {
        let non_terminal = 1.0f32 - dones[bi];
        let offset = bi * d;
        for di in 0..d {
            let target = features[offset + di] + gamma * non_terminal * psi_next[offset + di];
            out[offset + di] = target - psi_current[offset + di];
        }
    }
    out
}

/// Successor Features TD Error WGSL compute shader
pub const SUCCESSOR_FEATURES_TD_ERROR_WGSL: &str = r#"
struct Config {
    batch: u32,
    d: u32,
    gamma: f32,
};

@group(0) @binding(0) var<storage, read> psi_current: array<f32>;
@group(0) @binding(1) var<storage, read> features: array<f32>;
@group(0) @binding(2) var<storage, read> psi_next: array<f32>;
@group(0) @binding(3) var<storage, read> dones: array<f32>;
@group(0) @binding(4) var<storage, read_write> td_error: array<f32>;
@group(0) @binding(5) var<uniform> cfg: Config;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = cfg.batch * cfg.d;
    if (idx >= total) {
        return;
    }

    let bi = idx / cfg.d;
    let non_terminal = 1.0 - dones[bi];
    let target = features[idx] + cfg.gamma * non_terminal * psi_next[idx];
    td_error[idx] = target - psi_current[idx];
}
"#;
