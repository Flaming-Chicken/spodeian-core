//! CPU and WGSL kernels for PEFT Adapters Ia3 Prefix Prompt and Llama Adapter.

/// WGSL compute shader for fused LLaMA-Adapter zero-gating: y = seq_out + gate * prefix_out.
pub const FUSED_LLAMA_ADAPTER_GATE_WGSL: &str = r#"
struct Uniforms {
    total_elements: u32,
    gate: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> seq_out: array<f32>;
@group(0) @binding(2) var<storage, read> prefix_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn llama_adapter_gate_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= u.total_elements) { return; }
    y[idx] = seq_out[idx] + u.gate * prefix_out[idx];
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

/// CPU (IA)^3 multiplicative rescaling: `y[b, s, d] = activation[b, s, d] * scale[d]`.
pub fn ia3_rescale_cpu(
    activation: &[f32],
    scale: &[f32],
    batch: usize,
    seq_len: usize,
    dim: usize,
) -> Vec<f32> {
    let mut y = vec![0.0f32; batch * seq_len * dim];
    for b in 0..batch {
        for s in 0..seq_len {
            let offset = (b * seq_len + s) * dim;
            for d in 0..dim {
                y[offset + d] = activation[offset + d] * scale[d];
            }
        }
    }
    y
}

/// CPU LLaMA-Adapter zero-gate forward pass: `y = seq_out + gate * prefix_out`.
pub fn llama_adapter_zero_gate_cpu(seq_out: &[f32], prefix_out: &[f32], gate: f32) -> Vec<f32> {
    let n = seq_out.len();
    let mut y = vec![0.0f32; n];
    for i in 0..n {
        y[i] = seq_out[i] + gate * prefix_out[i];
    }
    y
}
