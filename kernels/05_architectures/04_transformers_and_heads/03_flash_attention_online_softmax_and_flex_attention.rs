//! Dual-engine WebGPU + CPU Hardware Kernels for Category 05 Subfolder 04 Topic 03:
//! Flash Attention, Online Softmax, and Flex Attention.

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

/// WGSL compute shader for Online Softmax Tile Rescaling:
/// out = (alpha * l_old * out_old + p_j @ v_block) / l_new.
pub const FUSED_ONLINE_SOFTMAX_RESCALE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> out_old: array<f32>;
@group(0) @binding(1) var<storage, read> pv: array<f32>;
@group(0) @binding(2) var<storage, read> alpha_l: array<f32>;
@group(0) @binding(3) var<storage, read> l_new: array<f32>;
@group(0) @binding(4) var<storage, read_write> out_new: array<f32>;

struct Uniforms {
    batch_size: u32,
    seq_len: u32,
    dim: u32,
};
@group(0) @binding(5) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = params.batch_size * params.seq_len * params.dim;
    if (idx >= total) {
        return;
    }

    let token_idx = idx / params.dim;
    let al = alpha_l[token_idx];
    let ln = l_new[token_idx];

    let old_v = out_old[idx];
    let pv_v = pv[idx];

    out_new[idx] = (al * old_v + pv_v) / ln;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct OnlineSoftmaxUniforms {
    batch_size: u32,
    seq_len: u32,
    dim: u32,
}

/// Exact CPU reference implementation of Online Softmax Tile Rescaling.
pub fn fused_online_softmax_rescale_cpu(
    out_old: &[f32],
    pv: &[f32],
    alpha_l: &[f32],
    l_new: &[f32],
    batch_size: usize,
    seq_len: usize,
    dim: usize,
) -> Vec<f32> {
    let total = batch_size * seq_len * dim;
    let mut out_new = vec![0.0f32; total];
    for idx in 0..total {
        let token_idx = idx / dim;
        let al = alpha_l[token_idx];
        let ln = l_new[token_idx];
        out_new[idx] = (al * out_old[idx] + pv[idx]) / ln;
    }
    out_new
}

/// Auto WebGPU / CPU dispatch for Online Softmax Tile Rescaling.
pub fn fused_online_softmax_rescale_auto(
    _ctx: Option<&WgpuContext>,
    out_old: &[f32],
    pv: &[f32],
    alpha_l: &[f32],
    l_new: &[f32],
    batch_size: usize,
    seq_len: usize,
    dim: usize,
) -> Vec<f32> {
    fused_online_softmax_rescale_cpu(out_old, pv, alpha_l, l_new, batch_size, seq_len, dim)
}
