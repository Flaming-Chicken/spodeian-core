//! CPU and WGSL kernels for DiLoCo Local SGD, PowerSGD, and Distributed Checkpointing.

/// WGSL compute shader for fused DiLoCo outer optimizer update:
/// `v[i] = beta * v[i] + delta[i]`, `theta[i] = theta[i] - lr * v[i]`.
pub const FUSED_DILOCO_OUTER_STEP_WGSL: &str = r#"
struct Uniforms {
    size: u32,
    lr: f32,
    beta: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> delta: array<f32>;        // [size]
@group(0) @binding(2) var<storage, read_write> v: array<f32>;      // [size]
@group(0) @binding(3) var<storage, read_write> theta: array<f32>;  // [size]

@compute @workgroup_size(64, 1, 1)
fn diloco_outer_step_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.size) { return; }

    let v_new = u.beta * v[i] + delta[i];
    v[i] = v_new;
    theta[i] = theta[i] - u.lr * v_new;
}
"#;

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

/// CPU DiLoCo outer step kernel: updates momentum and parameters in place.
pub fn diloco_outer_step_cpu(theta: &mut [f32], v: &mut [f32], delta: &[f32], lr: f32, beta: f32) {
    for i in 0..theta.len() {
        let v_new = beta * v[i] + delta[i];
        v[i] = v_new;
        theta[i] -= lr * v_new;
    }
}

/// CPU PowerSGD reconstruction kernel: `recon[i, j] = sum_r (p[i, r] * q[j, r])`.
pub fn powersgd_recon_cpu(p: &[f32], q: &[f32], m: usize, n: usize, rank: usize) -> Vec<f32> {
    let mut recon = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for r in 0..rank {
                acc += p[i * rank + r] * q[j * rank + r];
            }
            recon[i * n + j] = acc;
        }
    }
    recon
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diloco_outer_step_cpu() {
        let mut theta = vec![1.0, 2.0];
        let mut v = vec![0.0, 0.0];
        let delta = vec![0.5, 1.0];
        diloco_outer_step_cpu(&mut theta, &mut v, &delta, 0.5, 0.9);
        // v = 0.9 * 0 + delta = [0.5, 1.0]
        // theta = theta - 0.5 * v = [1.0 - 0.25, 2.0 - 0.5] = [0.75, 1.5]
        assert_eq!(v, vec![0.5, 1.0]);
        assert_eq!(theta, vec![0.75, 1.5]);
    }

    #[test]
    fn test_powersgd_recon_cpu() {
        let p = vec![1.0, 2.0]; // [2, 1]
        let q = vec![3.0, 4.0]; // [2, 1]
        let recon = powersgd_recon_cpu(&p, &q, 2, 2, 1);
        assert_eq!(recon, vec![3.0, 4.0, 6.0, 8.0]);
    }
}
