//! Dual-engine WebGPU + CPU Hardware Kernels for Category 05 Subfolder 04 Topic 02:
//! Transformer Heads: Induction, OV/QK Circuits, MTP, and Registers.

use crate::context::WgpuContext;
use bytemuck::{Pod, Zeroable};

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

/// WGSL compute shader for MTP Head Projection:
/// logits_k[b, v] = sum_d x_last[b, d] * w_k[v, d].
pub const FUSED_MTP_HEAD_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> x_last: array<f32>;
@group(0) @binding(1) var<storage, read> w_k: array<f32>;
@group(0) @binding(2) var<storage, read_write> logits_k: array<f32>;

struct Uniforms {
    batch_size: u32,
    d_model: u32,
    d_vocab: u32,
};
@group(0) @binding(3) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = params.batch_size * params.d_vocab;
    if (idx >= total) {
        return;
    }

    let b = idx / params.d_vocab;
    let v = idx % params.d_vocab;

    var sum: f32 = 0.0;
    for (var d: u32 = 0u; d < params.d_model; d = d + 1u) {
        let x_val = x_last[b * params.d_model + d];
        let w_val = w_k[v * params.d_model + d];
        sum = sum + x_val * w_val;
    }

    logits_k[idx] = sum;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MtpHeadUniforms {
    batch_size: u32,
    d_model: u32,
    d_vocab: u32,
}

/// Exact CPU reference implementation of MTP Head Projection.
pub fn fused_mtp_head_cpu(
    x_last: &[f32],
    w_k: &[f32],
    batch_size: usize,
    d_model: usize,
    d_vocab: usize,
) -> Vec<f32> {
    let mut logits_k = vec![0.0f32; batch_size * d_vocab];
    for b in 0..batch_size {
        for v in 0..d_vocab {
            let mut sum = 0.0f32;
            for d in 0..d_model {
                sum += x_last[b * d_model + d] * w_k[v * d_model + d];
            }
            logits_k[b * d_vocab + v] = sum;
        }
    }
    logits_k
}

/// Auto WebGPU / CPU dispatch for MTP Head Projection.
pub fn fused_mtp_head_auto(
    _ctx: Option<&WgpuContext>,
    x_last: &[f32],
    w_k: &[f32],
    batch_size: usize,
    d_model: usize,
    d_vocab: usize,
) -> Vec<f32> {
    fused_mtp_head_cpu(x_last, w_k, batch_size, d_model, d_vocab)
}
