//! CPU and WGSL kernels for Federated Learning Fedavg Fedprox Scaffold and Secagg.

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

/// Fused FedProx proximal gradient update CPU kernel:
/// w_next[i] = w_local[i] - eta * (grad_local[i] + mu * (w_local[i] - w_global[i])).
pub fn fedprox_gradient_step_cpu(
    w_local: &[f32],
    w_global: &[f32],
    grad_local: &[f32],
    d: usize,
    eta: f32,
    mu: f32,
) -> Vec<f32> {
    let mut w_next = vec![0.0f32; d];
    for i in 0..d {
        let diff = w_local[i] - w_global[i];
        let prox_g = grad_local[i] + mu * diff;
        w_next[i] = w_local[i] - eta * prox_g;
    }
    w_next
}

/// Fused FedProx proximal gradient update WGSL compute shader
pub const FEDPROX_PROXIMAL_STEP_WGSL: &str = r#"
struct Config {
    d: u32,
    eta: f32,
    mu: f32,
};

@group(0) @binding(0) var<storage, read> w_local: array<f32>;
@group(0) @binding(1) var<storage, read> w_global: array<f32>;
@group(0) @binding(2) var<storage, read> grad_local: array<f32>;
@group(0) @binding(3) var<storage, read_write> w_next: array<f32>;
@group(0) @binding(4) var<uniform> cfg: Config;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= cfg.d) {
        return;
    }

    let diff = w_local[i] - w_global[i];
    let prox_g = grad_local[i] + cfg.mu * diff;
    w_next[i] = w_local[i] - cfg.eta * prox_g;
}
"#;

/// Dispatcher: fallback to CPU when GPU context is not available
pub fn fedprox_gradient_step_auto(
    w_local: &[f32],
    w_global: &[f32],
    grad_local: &[f32],
    d: usize,
    eta: f32,
    mu: f32,
) -> Vec<f32> {
    fedprox_gradient_step_cpu(w_local, w_global, grad_local, d, eta, mu)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fedprox_gradient_step_cpu() {
        let wl = vec![1.0, 2.0];
        let wg = vec![0.0, 0.0];
        let gl = vec![0.1, 0.2];
        // prox_g = gl + 0.1 * wl = [0.1 + 0.1, 0.2 + 0.2] = [0.2, 0.4]
        // w_next = wl - 0.5 * prox_g = [1.0 - 0.1, 2.0 - 0.2] = [0.9, 1.8]
        let next = fedprox_gradient_step_cpu(&wl, &wg, &gl, 2, 0.5, 0.1);
        assert!((next[0] - 0.9).abs() < 1e-5);
        assert!((next[1] - 1.8).abs() < 1e-5);
    }
}
