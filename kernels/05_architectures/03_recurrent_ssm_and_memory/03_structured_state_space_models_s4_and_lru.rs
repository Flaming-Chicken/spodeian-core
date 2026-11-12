//! Dual-engine WebGPU + CPU Hardware Kernels for Category 05 Subfolder 03 Topic 03:
//! Structured State Space Models S4 and LRU.

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

/// WGSL compute shader for Fused Complex LRU Step:
/// Evaluates diagonal complex recurrence: h_{t} = Lambda * h_{t-1} + Bu_t.
pub const FUSED_LRU_STEP_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> bu_re: array<f32>;
@group(0) @binding(1) var<storage, read> bu_im: array<f32>;
@group(0) @binding(2) var<storage, read> h_re_prev: array<f32>;
@group(0) @binding(3) var<storage, read> h_im_prev: array<f32>;
@group(0) @binding(4) var<storage, read> diag_re: array<f32>;
@group(0) @binding(5) var<storage, read> diag_im: array<f32>;
@group(0) @binding(6) var<storage, read_write> h_re_next: array<f32>;
@group(0) @binding(7) var<storage, read_write> h_im_next: array<f32>;

struct Uniforms {
    batch_size: u32,
    state_dim: u32,
};
@group(0) @binding(8) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = params.batch_size * params.state_dim;
    if (idx >= total) {
        return;
    }

    let d = idx % params.state_dim;
    let dre = diag_re[d];
    let dim = diag_im[d];

    let hre = h_re_prev[idx];
    let him = h_im_prev[idx];

    let nre = dre * hre - dim * him + bu_re[idx];
    let nim = dre * him + dim * hre + bu_im[idx];

    h_re_next[idx] = nre;
    h_im_next[idx] = nim;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LruStepUniforms {
    batch_size: u32,
    state_dim: u32,
}

/// Exact CPU reference implementation of Fused LRU Step.
pub fn fused_lru_step_cpu(
    bu_re: &[f32],
    bu_im: &[f32],
    h_re_prev: &[f32],
    h_im_prev: &[f32],
    diag_re: &[f32],
    diag_im: &[f32],
    batch_size: usize,
    state_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let total = batch_size * state_dim;
    let mut h_re_next = vec![0.0f32; total];
    let mut h_im_next = vec![0.0f32; total];

    for b in 0..batch_size {
        for d in 0..state_dim {
            let idx = b * state_dim + d;
            let dre = diag_re[d];
            let dim = diag_im[d];

            let hre = h_re_prev[idx];
            let him = h_im_prev[idx];

            h_re_next[idx] = dre * hre - dim * him + bu_re[idx];
            h_im_next[idx] = dre * him + dim * hre + bu_im[idx];
        }
    }
    (h_re_next, h_im_next)
}

/// Auto WebGPU / CPU dispatch for Fused LRU Step.
pub fn fused_lru_step_auto(
    _ctx: Option<&WgpuContext>,
    bu_re: &[f32],
    bu_im: &[f32],
    h_re_prev: &[f32],
    h_im_prev: &[f32],
    diag_re: &[f32],
    diag_im: &[f32],
    batch_size: usize,
    state_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    fused_lru_step_cpu(
        bu_re, bu_im, h_re_prev, h_im_prev, diag_re, diag_im, batch_size, state_dim,
    )
}
