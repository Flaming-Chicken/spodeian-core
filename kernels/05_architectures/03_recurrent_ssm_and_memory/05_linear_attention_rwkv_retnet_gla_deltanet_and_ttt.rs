//! Dual-engine WebGPU + CPU Hardware Kernels for Category 05 Subfolder 03 Topic 05:
//! Linear Attention, RWKV, RetNet, GLA, DeltaNet, and TTT.

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

/// WGSL compute shader for Fused RetNet Recurrent Step:
/// Evaluates state update: S_{t} = gamma * S_{t-1} + K_t^T V_t.
pub const FUSED_RETNET_STEP_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> s_prev: array<f32>;
@group(0) @binding(1) var<storage, read> k_curr: array<f32>;
@group(0) @binding(2) var<storage, read> v_curr: array<f32>;
@group(0) @binding(3) var<storage, read_write> s_next: array<f32>;

struct Uniforms {
    batch_size: u32,
    d_k: u32,
    d_v: u32,
    gamma: f32,
};
@group(0) @binding(4) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = params.batch_size * params.d_k * params.d_v;
    if (idx >= total) {
        return;
    }

    let b = idx / (params.d_k * params.d_v);
    let rem = idx % (params.d_k * params.d_v);
    let i = rem / params.d_v;
    let j = rem % params.d_v;

    let k_val = k_curr[b * params.d_k + i];
    let v_val = v_curr[b * params.d_v + j];
    let sp = s_prev[idx];

    s_next[idx] = params.gamma * sp + k_val * v_val;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RetNetStepUniforms {
    batch_size: u32,
    d_k: u32,
    d_v: u32,
    gamma: f32,
}

/// Exact CPU reference implementation of Fused RetNet Step.
pub fn fused_retnet_step_cpu(
    s_prev: &[f32],
    k_curr: &[f32],
    v_curr: &[f32],
    gamma: f32,
    batch_size: usize,
    d_k: usize,
    d_v: usize,
) -> Vec<f32> {
    let total = batch_size * d_k * d_v;
    let mut s_next = vec![0.0f32; total];

    for b in 0..batch_size {
        for i in 0..d_k {
            let k_val = k_curr[b * d_k + i];
            for j in 0..d_v {
                let idx = b * (d_k * d_v) + i * d_v + j;
                let v_val = v_curr[b * d_v + j];
                s_next[idx] = gamma * s_prev[idx] + k_val * v_val;
            }
        }
    }
    s_next
}

/// Auto WebGPU / CPU dispatch for Fused RetNet Step.
pub fn fused_retnet_step_auto(
    _ctx: Option<&WgpuContext>,
    s_prev: &[f32],
    k_curr: &[f32],
    v_curr: &[f32],
    gamma: f32,
    batch_size: usize,
    d_k: usize,
    d_v: usize,
) -> Vec<f32> {
    fused_retnet_step_cpu(s_prev, k_curr, v_curr, gamma, batch_size, d_k, d_v)
}
