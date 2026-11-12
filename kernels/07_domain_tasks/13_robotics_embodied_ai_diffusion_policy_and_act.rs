//! CPU and WGSL kernels for Robotics Embodied Ai Diffusion Policy and Act.

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

/// Fused ACT Temporal Ensemble Action Reduction:
/// action_preds: [B, K, D_a] row-major -> index = (bi * K + ki) * D_a + di
/// weights: [K] normalized weights sum_k w_k = 1.0
/// out: [B, D_a] row-major -> index = bi * D_a + di
pub fn temporal_ensemble_cpu(
    action_preds: &[f32],
    weights: &[f32],
    b: usize,
    k: usize,
    d_a: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; b * d_a];
    for bi in 0..b {
        for di in 0..d_a {
            let mut sum = 0.0f32;
            for ki in 0..k {
                let idx = (bi * k + ki) * d_a + di;
                sum += action_preds[idx] * weights[ki];
            }
            out[bi * d_a + di] = sum;
        }
    }
    out
}

/// Fused ACT Temporal Ensemble Action WGSL compute shader
pub const TEMPORAL_ENSEMBLE_WGSL: &str = r#"
struct Config {
    b: u32,
    k: u32,
    d_a: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> action_preds: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> blended: array<f32>;
@group(0) @binding(3) var<uniform> cfg: Config;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let di = global_id.x;
    let bi = global_id.y;

    if (bi >= cfg.b || di >= cfg.d_a) {
        return;
    }

    var sum: f32 = 0.0;
    for (var ki: u32 = 0u; ki < cfg.k; ki = ki + 1u) {
        let idx = (bi * cfg.k + ki) * cfg.d_a + di;
        sum = sum + action_preds[idx] * weights[ki];
    }

    blended[bi * cfg.d_a + di] = sum;
}
"#;

/// Dispatcher: fallback to CPU when GPU context is not available
pub fn temporal_ensemble_auto(
    action_preds: &[f32],
    weights: &[f32],
    b: usize,
    k: usize,
    d_a: usize,
) -> Vec<f32> {
    temporal_ensemble_cpu(action_preds, weights, b, k, d_a)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_ensemble_cpu() {
        let actions = vec![1.0, 2.0, 3.0, 4.0]; // [1, 2, 2]
        let weights = vec![0.5, 0.5];
        let res = temporal_ensemble_cpu(&actions, &weights, 1, 2, 2);
        assert_eq!(res.len(), 2);
        // (1.0 * 0.5 + 3.0 * 0.5) = 2.0
        // (2.0 * 0.5 + 4.0 * 0.5) = 3.0
        assert!((res[0] - 2.0).abs() < 1e-5);
        assert!((res[1] - 3.0).abs() < 1e-5);
    }
}
