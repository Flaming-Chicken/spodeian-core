//! CPU (and optional WGSL) kernels for Profiling Roofline Analysis CUDA Graphs and Fused Optimizers.

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

/// WGSL compute shader for single-pass fused AdamW optimizer update.
pub const FUSED_ADAMW_STEP_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> w_in: array<f32>;
@group(0) @binding(1) var<storage, read> g_in: array<f32>;
@group(0) @binding(2) var<storage, read> m_in: array<f32>;
@group(0) @binding(3) var<storage, read> v_in: array<f32>;
@group(0) @binding(4) var<storage, read_write> w_out: array<f32>;
@group(0) @binding(5) var<storage, read_write> m_out: array<f32>;
@group(0) @binding(6) var<storage, read_write> v_out: array<f32>;

struct AdamParams {
    total_elements: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    weight_decay: f32,
    eps: f32,
    bias_corr1: f32,
    bias_corr2: f32,
}
@group(0) @binding(7) var<uniform> params: AdamParams;

@compute @workgroup_size(64)
fn fused_adamw_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.total_elements) {
        return;
    }

    let g = g_in[idx];
    let mt = params.beta1 * m_in[idx] + (1.0 - params.beta1) * g;
    let vt = params.beta2 * v_in[idx] + (1.0 - params.beta2) * (g * g);

    m_out[idx] = mt;
    v_out[idx] = vt;

    let m_hat = mt / params.bias_corr1;
    let v_hat = vt / params.bias_corr2;

    let w_decayed = w_in[idx] * (1.0 - params.lr * params.weight_decay);
    w_out[idx] = w_decayed - params.lr * (m_hat / (sqrt(v_hat) + params.eps));
}
"#;

/// CPU reference implementation for fused AdamW update step.
pub fn fused_adamw_step_cpu(
    w: &[f32],
    g: &[f32],
    m: &[f32],
    v: &[f32],
    step: usize,
    lr: f32,
    beta1: f32,
    beta2: f32,
    weight_decay: f32,
    eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let t = step as f32;
    let bias_corr1 = 1.0f32 - beta1.powf(t);
    let bias_corr2 = 1.0f32 - beta2.powf(t);

    let n = w.len();
    let mut w_out = Vec::with_capacity(n);
    let mut m_out = Vec::with_capacity(n);
    let mut v_out = Vec::with_capacity(n);

    for i in 0..n {
        let gv = g[i];
        let mt = beta1 * m[i] + (1.0f32 - beta1) * gv;
        let vt = beta2 * v[i] + (1.0f32 - beta2) * (gv * gv);

        let m_hat = mt / bias_corr1;
        let v_hat = vt / bias_corr2;

        let w_decayed = w[i] * (1.0f32 - lr * weight_decay);
        let wt = w_decayed - lr * (m_hat / (v_hat.sqrt() + eps));

        w_out.push(wt);
        m_out.push(mt);
        v_out.push(vt);
    }

    (w_out, m_out, v_out)
}
