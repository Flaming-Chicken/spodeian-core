//! CPU and WGSL kernels for Zero Stages 1 2 3 and FSDP2 Fully Shard.

/// WGSL compute shader for fused in-place Adam optimizer update on sharded parameters.
pub const FUSED_ZERO_ADAM_UPDATE_WGSL: &str = r#"
struct Uniforms {
    num_params: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    bias_corr1: f32,
    bias_corr2: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> g: array<f32>;
@group(0) @binding(2) var<storage, read_write> p: array<f32>;
@group(0) @binding(3) var<storage, read_write> m: array<f32>;
@group(0) @binding(4) var<storage, read_write> v: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn zero_adam_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= u.num_params) { return; }

    let grad = g[idx];
    let m_val = u.beta1 * m[idx] + (1.0 - u.beta1) * grad;
    let v_val = u.beta2 * v[idx] + (1.0 - u.beta2) * (grad * grad);

    m[idx] = m_val;
    v[idx] = v_val;

    let m_hat = m_val / u.bias_corr1;
    let v_hat = v_val / u.bias_corr2;

    p[idx] = p[idx] - u.lr * m_hat / (sqrt(v_hat) + u.eps);
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

/// CPU in-place Adam optimizer update on sharded parameters.
pub fn zero_adam_step_cpu(
    p: &mut [f32],
    g: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    step: usize,
) {
    let bias_corr1 = 1.0f32 - beta1.powi(step as i32);
    let bias_corr2 = 1.0f32 - beta2.powi(step as i32);

    for i in 0..p.len() {
        let grad = g[i];
        let m_val = beta1 * m[i] + (1.0f32 - beta1) * grad;
        let v_val = beta2 * v[i] + (1.0f32 - beta2) * (grad * grad);

        m[i] = m_val;
        v[i] = v_val;

        let m_hat = m_val / bias_corr1;
        let v_hat = v_val / bias_corr2;

        p[i] -= lr * m_hat / (v_hat.sqrt() + eps);
    }
}
