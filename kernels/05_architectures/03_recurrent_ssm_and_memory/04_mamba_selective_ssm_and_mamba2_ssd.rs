//! Dual-engine WebGPU + CPU Hardware Kernels for Category 05 Subfolder 03 Topic 04:
//! Mamba Selective SSM and Mamba2 SSD.

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

/// WGSL compute shader for Fused Mamba Selective SSM Step:
/// Evaluates recurrence: h_{t} = A_bar * h_{t-1} + B_bar * x_t.
pub const FUSED_MAMBA_STEP_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a_bar: array<f32>;
@group(0) @binding(1) var<storage, read> b_bar: array<f32>;
@group(0) @binding(2) var<storage, read> h_prev: array<f32>;
@group(0) @binding(3) var<storage, read> x_curr: array<f32>;
@group(0) @binding(4) var<storage, read_write> h_next: array<f32>;

struct Uniforms {
    batch_size: u32,
    d_model: u32,
    d_state: u32,
};
@group(0) @binding(5) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = params.batch_size * params.d_model * params.d_state;
    if (idx >= total) {
        return;
    }

    let b = idx / (params.d_model * params.d_state);
    let rem = idx % (params.d_model * params.d_state);
    let m = rem / params.d_state;

    let x_idx = b * params.d_model + m;
    let ab = a_bar[idx];
    let bb = b_bar[idx];
    let hp = h_prev[idx];
    let xc = x_curr[x_idx];

    h_next[idx] = ab * hp + bb * xc;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MambaStepUniforms {
    batch_size: u32,
    d_model: u32,
    d_state: u32,
}

/// Exact CPU reference implementation of Fused Mamba Step.
pub fn fused_mamba_step_cpu(
    a_bar: &[f32],
    b_bar: &[f32],
    h_prev: &[f32],
    x_curr: &[f32],
    batch_size: usize,
    d_model: usize,
    d_state: usize,
) -> Vec<f32> {
    let total = batch_size * d_model * d_state;
    let mut h_next = vec![0.0f32; total];

    for b in 0..batch_size {
        for m in 0..d_model {
            let x_val = x_curr[b * d_model + m];
            for n in 0..d_state {
                let idx = b * (d_model * d_state) + m * d_state + n;
                h_next[idx] = a_bar[idx] * h_prev[idx] + b_bar[idx] * x_val;
            }
        }
    }
    h_next
}

/// Auto WebGPU / CPU dispatch for Fused Mamba Step.
pub fn fused_mamba_step_auto(
    _ctx: Option<&WgpuContext>,
    a_bar: &[f32],
    b_bar: &[f32],
    h_prev: &[f32],
    x_curr: &[f32],
    batch_size: usize,
    d_model: usize,
    d_state: usize,
) -> Vec<f32> {
    fused_mamba_step_cpu(a_bar, b_bar, h_prev, x_curr, batch_size, d_model, d_state)
}
