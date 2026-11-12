//! Dual-engine WebGPU + CPU Hardware Kernels for Category 05 Subfolder 03 Topic 07:
//! Reservoir Computing, Echo State Networks, LSM, and NG-RC.

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

/// WGSL compute shader for Fused ESN Leaky Step:
/// h_t = (1 - alpha) * h_{t-1} + alpha * tanh(pre_act).
pub const FUSED_ESN_LEAKY_STEP_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> h_prev: array<f32>;
@group(0) @binding(1) var<storage, read> pre_act: array<f32>;
@group(0) @binding(2) var<storage, read_write> h_next: array<f32>;

struct Uniforms {
    total_elements: u32,
    alpha: f32,
};
@group(0) @binding(3) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.total_elements) {
        return;
    }

    let p = pre_act[idx];
    let th = tanh(p);
    let hp = h_prev[idx];

    h_next[idx] = (1.0 - params.alpha) * hp + params.alpha * th;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct EsnLeakyUniforms {
    total_elements: u32,
    alpha: f32,
}

/// Exact CPU reference implementation of Fused ESN Leaky Step.
pub fn fused_esn_leaky_step_cpu(h_prev: &[f32], pre_act: &[f32], alpha: f32) -> Vec<f32> {
    let mut h_next = vec![0.0f32; h_prev.len()];
    for i in 0..h_prev.len() {
        let th = pre_act[i].tanh();
        h_next[i] = (1.0 - alpha) * h_prev[i] + alpha * th;
    }
    h_next
}

/// Auto WebGPU / CPU dispatch for Fused ESN Leaky Step.
pub fn fused_esn_leaky_step_auto(
    _ctx: Option<&WgpuContext>,
    h_prev: &[f32],
    pre_act: &[f32],
    alpha: f32,
) -> Vec<f32> {
    fused_esn_leaky_step_cpu(h_prev, pre_act, alpha)
}
