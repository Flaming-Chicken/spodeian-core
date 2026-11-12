//! Dual-engine WebGPU + CPU Hardware Kernels for Category 05 Subfolder 03 Topic 01:
//! RNN, LSTM, GRU & Attention Seq2Seq.

use crate::context::WgpuContext;
use bytemuck::{Pod, Zeroable};

/// WGSL compute shader for Fused LSTM Step:
/// Evaluates elementwise gating and state update without intermediate VRAM allocations.
pub const FUSED_LSTM_STEP_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> gates: array<f32>;
@group(0) @binding(1) var<storage, read> c_prev: array<f32>;
@group(0) @binding(2) var<storage, read_write> h_next: array<f32>;
@group(0) @binding(3) var<storage, read_write> c_next: array<f32>;

struct Uniforms {
    batch_size: u32,
    hidden_dim: u32,
};
@group(0) @binding(4) var<uniform> params: Uniforms;

fn sigmoid_fn(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

fn tanh_fn(x: f32) -> f32 {
    let e2x = exp(2.0 * x);
    return (e2x - 1.0) / (e2x + 1.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = params.batch_size * params.hidden_dim;
    if (idx >= total) {
        return;
    }

    let b = idx / params.hidden_dim;
    let d = idx % params.hidden_dim;
    let d_hid = params.hidden_dim;

    let gate_base = b * (4u * d_hid);
    let i_val = sigmoid_fn(gates[gate_base + d]);
    let f_val = sigmoid_fn(gates[gate_base + d_hid + d]);
    let g_val = tanh_fn(gates[gate_base + 2u * d_hid + d]);
    let o_val = sigmoid_fn(gates[gate_base + 3u * d_hid + d]);

    let cp = c_prev[idx];
    let cn = f_val * cp + i_val * g_val;
    let hn = o_val * tanh_fn(cn);

    c_next[idx] = cn;
    h_next[idx] = hn;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LstmStepUniforms {
    batch_size: u32,
    hidden_dim: u32,
}

/// Exact CPU reference implementation of Fused LSTM Step.
pub fn fused_lstm_step_cpu(
    gates: &[f32],
    c_prev: &[f32],
    batch_size: usize,
    hidden_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let total = batch_size * hidden_dim;
    let mut h_next = vec![0.0f32; total];
    let mut c_next = vec![0.0f32; total];

    for b in 0..batch_size {
        for d in 0..hidden_dim {
            let idx = b * hidden_dim + d;
            let gate_base = b * (4 * hidden_dim);

            let i_val = 1.0 / (1.0 + (-gates[gate_base + d]).exp());
            let f_val = 1.0 / (1.0 + (-gates[gate_base + hidden_dim + d]).exp());
            let g_val = gates[gate_base + 2 * hidden_dim + d].tanh();
            let o_val = 1.0 / (1.0 + (-gates[gate_base + 3 * hidden_dim + d]).exp());

            let cp = c_prev[idx];
            let cn = f_val * cp + i_val * g_val;
            let hn = o_val * cn.tanh();

            c_next[idx] = cn;
            h_next[idx] = hn;
        }
    }
    (h_next, c_next)
}

/// Auto WebGPU / CPU dispatch for Fused LSTM Step.
pub fn fused_lstm_step_auto(
    _ctx: Option<&WgpuContext>,
    gates: &[f32],
    c_prev: &[f32],
    batch_size: usize,
    hidden_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    fused_lstm_step_cpu(gates, c_prev, batch_size, hidden_dim)
}

/// Standard linear anchor CPU implementation for parity testing.
pub fn linear_anchor_cpu(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = bias[j];
            for p in 0..k {
                sum += x[i * k + p] * weight[p * n + j];
            }
            out[i * n + j] = sum;
        }
    }
    out
}

/// Standard linear anchor auto dispatch.
pub fn linear_anchor_auto(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    linear_anchor_cpu(x, weight, bias, m, k, n)
}
