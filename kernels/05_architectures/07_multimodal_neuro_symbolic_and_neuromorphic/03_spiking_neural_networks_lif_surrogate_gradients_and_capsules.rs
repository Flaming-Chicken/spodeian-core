//! CPU (and optional WGSL) kernels for Spiking Neural Networks LIF Surrogate Gradients and Capsules.

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

/// Single-step CPU kernel for Leaky Integrate-and-Fire neuron: v_new, s_new.
pub fn lif_step_cpu(
    v_prev: &[f32],
    s_prev: &[f32],
    current: &[f32],
    beta: f32,
    v_th: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = v_prev.len();
    let mut v_new = Vec::with_capacity(n);
    let mut s_new = Vec::with_capacity(n);

    for i in 0..n {
        let v = beta * v_prev[i] * (1.0 - s_prev[i]) + current[i];
        let s = if v >= v_th { 1.0 } else { 0.0 };
        v_new.push(v);
        s_new.push(s);
    }
    (v_new, s_new)
}

/// WGSL shader for fused Leaky Integrate-and-Fire neuron update with hard reset.
pub const LIF_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> v_prev: array<f32>;
@group(0) @binding(1) var<storage, read> s_prev: array<f32>;
@group(0) @binding(2) var<storage, read> current: array<f32>;
@group(0) @binding(3) var<storage, read_write> v_new: array<f32>;
@group(0) @binding(4) var<storage, read_write> s_new: array<f32>;

struct Uniforms {
    total_elements: u32,
    beta: f32,
    v_th: f32,
};
@group(0) @binding(5) var<uniform> u: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= u.total_elements) {
        return;
    }
    let v = u.beta * v_prev[idx] * (1.0 - s_prev[idx]) + current[idx];
    let s = select(0.0, 1.0, v >= u.v_th);
    v_new[idx] = v;
    s_new[idx] = s;
}
"#;
