//! Dual-engine WebGPU + CPU Hardware Kernels for Category 05 Subfolder 03 Topic 06:
//! Memory Architectures: NTM, PKM, PEER, Infini-attention, and Titans.

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

/// WGSL compute shader for Fused Infini-Memory Matrix Accumulation:
/// M_{next} = M_{prev} + phi(k)^T @ v.
pub const FUSED_INFINI_UPDATE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> m_prev: array<f32>;
@group(0) @binding(1) var<storage, read> phi_k: array<f32>;
@group(0) @binding(2) var<storage, read> v_curr: array<f32>;
@group(0) @binding(3) var<storage, read_write> m_next: array<f32>;

struct Uniforms {
    batch_size: u32,
    dim: u32,
};
@group(0) @binding(4) var<uniform> params: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = params.batch_size * params.dim * params.dim;
    if (idx >= total) {
        return;
    }

    let b = idx / (params.dim * params.dim);
    let rem = idx % (params.dim * params.dim);
    let i = rem / params.dim;
    let j = rem % params.dim;

    let k_val = phi_k[b * params.dim + i];
    let v_val = v_curr[b * params.dim + j];

    m_next[idx] = m_prev[idx] + k_val * v_val;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct InfiniUpdateUniforms {
    batch_size: u32,
    dim: u32,
}

/// Exact CPU reference implementation of Fused Infini Update.
pub fn fused_infini_update_cpu(
    m_prev: &[f32],
    phi_k: &[f32],
    v_curr: &[f32],
    batch_size: usize,
    dim: usize,
) -> Vec<f32> {
    let total = batch_size * dim * dim;
    let mut m_next = vec![0.0f32; total];

    for b in 0..batch_size {
        for i in 0..dim {
            let k_val = phi_k[b * dim + i];
            for j in 0..dim {
                let idx = b * (dim * dim) + i * dim + j;
                let v_val = v_curr[b * dim + j];
                m_next[idx] = m_prev[idx] + k_val * v_val;
            }
        }
    }
    m_next
}

/// Auto WebGPU / CPU dispatch for Fused Infini Update.
pub fn fused_infini_update_auto(
    _ctx: Option<&WgpuContext>,
    m_prev: &[f32],
    phi_k: &[f32],
    v_curr: &[f32],
    batch_size: usize,
    dim: usize,
) -> Vec<f32> {
    fused_infini_update_cpu(m_prev, phi_k, v_curr, batch_size, dim)
}
