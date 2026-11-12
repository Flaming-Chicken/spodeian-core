//! CPU (and optional WGSL) kernels for Diffusion Conditioning Controlnet and Ip Adapter.

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

/// CPU kernel for IP-Adapter decoupled attention blending: `y = attn_text + lambda_img * attn_img`.
pub fn ip_adapter_blend_cpu(attn_text: &[f32], attn_img: &[f32], lambda_img: f32) -> Vec<f32> {
    attn_text
        .iter()
        .zip(attn_img.iter())
        .map(|(&t, &img)| t + lambda_img * img)
        .collect()
}

/// WGSL compute shader for fused IP-Adapter cross-attention accumulation.
pub const WGSL_IP_ADAPTER_BLEND: &str = r#"
@group(0) @binding(0) var<storage, read> attn_text: array<f32>;
@group(0) @binding(1) var<storage, read> attn_img: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> lambda_img: f32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&attn_text)) {
        return;
    }
    y[idx] = attn_text[idx] + lambda_img * attn_img[idx];
}
"#;
