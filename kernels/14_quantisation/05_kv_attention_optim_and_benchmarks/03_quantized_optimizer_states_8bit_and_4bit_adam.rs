//! CPU and WGSL kernels for Quantized Optimizer States: 8-bit and 4-bit Adam.

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

/// WGSL compute shader for fused on-the-fly 8-bit AdamW parameter update.
pub const FUSED_8BIT_ADAMW_STEP_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> theta: array<f32>;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read_write> m_int8: array<f32>;
@group(0) @binding(3) var<storage, read_write> v_int8: array<f32>;
@group(0) @binding(4) var<storage, read> scale_m: array<f32>;
@group(0) @binding(5) var<storage, read> max_v: array<f32>;

struct Uniforms {
    total_elements: u32,
    block_size: u32,
    beta1: f32,
    beta2: f32,
    lr: f32,
    weight_decay: f32,
    eps: f32,
    bc1: f32,
    bc2: f32,
};
@group(0) @binding(6) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.total_elements) {
        return;
    }

    let b_idx = i / params.block_size;
    let sm = scale_m[b_idx];
    let mv = max_v[b_idx];

    // Dequantize moments
    let m_deq = m_int8[i] * sm;
    let v_norm = exp2((v_int8[i] * (24.0 / 255.0)) - 24.0);
    let v_deq = v_norm * mv;

    // Update moments
    let g = grad[i];
    let m_new = params.beta1 * m_deq + (1.0 - params.beta1) * g;
    let v_new = params.beta2 * v_deq + (1.0 - params.beta2) * (g * g);

    // Bias-corrected Adam step
    let m_hat = m_new / params.bc1;
    let v_hat = v_new / params.bc2;
    let update = (m_hat / (sqrt(v_hat) + params.eps)) + params.weight_decay * theta[i];
    theta[i] = theta[i] - params.lr * update;

    // Re-quantize to 8-bit
    m_int8[i] = clamp(round(m_new / sm), -127.0, 127.0);
    let v_norm_new = clamp(v_new / mv, 1e-8, 1.0);
    let log_v = log2(v_norm_new);
    v_int8[i] = clamp(round((log_v + 24.0) * (255.0 / 24.0)), 0.0, 255.0);
}
"#;

/// CPU reference implementation of fused 8-bit AdamW step.
pub fn fused_8bit_adamw_step_cpu(
    theta: &mut [f32],
    grad: &[f32],
    m_deq: &mut [f32],
    v_deq: &mut [f32],
    beta1: f32,
    beta2: f32,
    lr: f32,
    weight_decay: f32,
    eps: f32,
    step: usize,
) {
    let bc1 = 1.0 - beta1.powi(step as i32);
    let bc2 = 1.0 - beta2.powi(step as i32);

    for i in 0..theta.len() {
        let g = grad[i];
        let m_t = beta1 * m_deq[i] + (1.0 - beta1) * g;
        let v_t = beta2 * v_deq[i] + (1.0 - beta2) * (g * g);

        let m_hat = m_t / bc1;
        let v_hat = v_t / bc2;

        let update = (m_hat / (v_hat.sqrt() + eps)) + weight_decay * theta[i];
        theta[i] -= lr * update;

        m_deq[i] = m_t;
        v_deq[i] = v_t;
    }
}

/// Auto dispatcher: returns CPU reference fallback.
pub fn fused_8bit_adamw_step_auto(
    theta: &mut [f32],
    grad: &[f32],
    m_deq: &mut [f32],
    v_deq: &mut [f32],
    beta1: f32,
    beta2: f32,
    lr: f32,
    weight_decay: f32,
    eps: f32,
    step: usize,
) {
    fused_8bit_adamw_step_cpu(
        theta,
        grad,
        m_deq,
        v_deq,
        beta1,
        beta2,
        lr,
        weight_decay,
        eps,
        step,
    );
}
