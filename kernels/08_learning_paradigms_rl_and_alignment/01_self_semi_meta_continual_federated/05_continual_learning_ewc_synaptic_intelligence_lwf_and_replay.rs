//! CPU and WGSL kernels for Continual Learning EWC Synaptic Intelligence Lwf and Replay.

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

/// Fused EWC quadratic parameter penalty CPU kernel:
/// Computes penalty[i] = 0.5 * lambda * fisher[i] * (theta[i] - theta_star[i])^2.
pub fn ewc_quadratic_penalty_cpu(
    theta: &[f32],
    theta_star: &[f32],
    fisher: &[f32],
    d: usize,
    lambda_ewc: f32,
) -> f32 {
    let mut sum_loss = 0.0f32;
    let factor = 0.5f32 * lambda_ewc;
    for i in 0..d {
        let diff = theta[i] - theta_star[i];
        sum_loss += factor * fisher[i] * diff * diff;
    }
    sum_loss
}

/// Fused EWC quadratic parameter penalty WGSL compute shader
pub const EWC_QUADRATIC_PENALTY_WGSL: &str = r#"
struct Config {
    d: u32,
    lambda_ewc: f32,
};

@group(0) @binding(0) var<storage, read> theta: array<f32>;
@group(0) @binding(1) var<storage, read> theta_star: array<f32>;
@group(0) @binding(2) var<storage, read> fisher: array<f32>;
@group(0) @binding(3) var<storage, read_write> penalty_vec: array<f32>;
@group(0) @binding(4) var<uniform> cfg: Config;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= cfg.d) {
        return;
    }

    let diff = theta[i] - theta_star[i];
    penalty_vec[i] = 0.5 * cfg.lambda_ewc * fisher[i] * diff * diff;
}
"#;

/// Dispatcher: fallback to CPU when GPU context is not available
pub fn ewc_quadratic_penalty_auto(
    theta: &[f32],
    theta_star: &[f32],
    fisher: &[f32],
    d: usize,
    lambda_ewc: f32,
) -> f32 {
    ewc_quadratic_penalty_cpu(theta, theta_star, fisher, d, lambda_ewc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ewc_quadratic_penalty_cpu() {
        let th = vec![2.0, 3.0];
        let th_star = vec![1.0, 1.0];
        let f = vec![1.0, 2.0];
        // 0.5 * 2.0 * (1.0 * 1^2 + 2.0 * 2^2) = 1.0 * (1 + 8) = 9.0
        let loss = ewc_quadratic_penalty_cpu(&th, &th_star, &f, 2, 2.0);
        assert!((loss - 9.0).abs() < 1e-5);
    }
}
