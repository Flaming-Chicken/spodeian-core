//! Topic 06: AI by Purpose - Prediction, Prescription, Control, Combinatorial & Symbolic (Dual Engine: WGSL + CPU fallback).

use crate::context::{KernelDevice, WgpuContext};

pub const WGSL_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

struct Dims {
    batch: u32,
    in_dim: u32,
    out_dim: u32,
};
@group(0) @binding(4) var<uniform> dims: Dims;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    let col = id.y;
    if (row >= dims.batch || col >= dims.out_dim) {
        return;
    }
    var acc: f32 = b[col];
    for (var i: u32 = 0u; i < dims.in_dim; i = i + 1u) {
        acc = acc + x[row * dims.in_dim + i] * w[i * dims.out_dim + col];
    }
    y[row * dims.out_dim + col] = acc;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Dims {
    batch: u32,
    in_dim: u32,
    out_dim: u32,
}

pub fn linear_forward_cpu(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    batch: usize,
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    let mut y = vec![0.0f32; batch * out_dim];
    for i in 0..batch {
        for j in 0..out_dim {
            let mut acc = b[j];
            for k in 0..in_dim {
                acc += x[i * in_dim + k] * w[k * out_dim + j];
            }
            y[i * out_dim + j] = acc;
        }
    }
    y
}

/// Evaluates cost prediction on CPU: softplus(ReLU(x @ w1 + b1) @ w2 + b2).
pub fn cost_predictor_forward_cpu(
    x: &[f32],
    w_cost1: &[f32],
    b_cost1: &[f32],
    w_cost2: &[f32],
    b_cost2: &[f32],
    batch: usize,
    in_dim: usize,
    hidden_dim: usize,
    cost_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * cost_dim];
    for i in 0..batch {
        let x_i = &x[i * in_dim..(i + 1) * in_dim];
        let mut h = vec![0.0f32; hidden_dim];
        for hd in 0..hidden_dim {
            let mut acc = b_cost1[hd];
            for k in 0..in_dim {
                acc += x_i[k] * w_cost1[k * hidden_dim + hd];
            }
            h[hd] = acc.max(0.0f32);
        }
        for j in 0..cost_dim {
            let mut acc = b_cost2[j];
            for hd in 0..hidden_dim {
                acc += h[hd] * w_cost2[hd * cost_dim + j];
            }
            // Softplus: ln(1 + exp(acc))
            out[i * cost_dim + j] = if acc > 20.0 {
                acc
            } else if acc < -20.0 {
                acc.exp()
            } else {
                (1.0f32 + acc.exp()).ln()
            };
        }
    }
    out
}

/// Solves simplex LP on CPU: one_hot(argmin_i c_i).
pub fn solve_lp_simplex_cpu(cost_vectors: &[f32], batch: usize, cost_dim: usize) -> Vec<f32> {
    let mut w_star = vec![0.0f32; batch * cost_dim];
    for i in 0..batch {
        let row = &cost_vectors[i * cost_dim..(i + 1) * cost_dim];
        let mut min_idx = 0;
        let mut min_val = f32::INFINITY;
        for c in 0..cost_dim {
            if row[c] < min_val {
                min_val = row[c];
                min_idx = c;
            }
        }
        w_star[i * cost_dim + min_idx] = 1.0f32;
    }
    w_star
}

/// Evaluates decision regret: c*^T w*(c_pred) - c*^T w*(c*).
pub fn decision_regret_cpu(
    c_pred: &[f32],
    c_star: &[f32],
    batch: usize,
    cost_dim: usize,
) -> Vec<f32> {
    let w_pred = solve_lp_simplex_cpu(c_pred, batch, cost_dim);
    let w_star = solve_lp_simplex_cpu(c_star, batch, cost_dim);

    let mut regret = vec![0.0f32; batch];
    for i in 0..batch {
        let mut cost_pred = 0.0f32;
        let mut cost_oracle = 0.0f32;
        for c in 0..cost_dim {
            let cs = c_star[i * cost_dim + c];
            cost_pred += cs * w_pred[i * cost_dim + c];
            cost_oracle += cs * w_star[i * cost_dim + c];
        }
        regret[i] = cost_pred - cost_oracle;
    }
    regret
}

/// Evaluates convex SPO+ loss on CPU.
pub fn spo_plus_loss_cpu(
    c_pred: &[f32],
    c_star: &[f32],
    batch: usize,
    cost_dim: usize,
) -> Vec<f32> {
    let w_star = solve_lp_simplex_cpu(c_star, batch, cost_dim);
    let mut loss = vec![0.0f32; batch];

    for i in 0..batch {
        let mut max_term = f32::NEG_INFINITY;
        let mut cost_oracle = 0.0f32;
        let mut cp_w_star = 0.0f32;

        for c in 0..cost_dim {
            let cp = c_pred[i * cost_dim + c];
            let cs = c_star[i * cost_dim + c];
            let ws = w_star[i * cost_dim + c];

            cost_oracle += cs * ws;
            cp_w_star += cp * ws;

            let v = cs - 2.0f32 * cp;
            if v > max_term {
                max_term = v;
            }
        }

        let lin_term = 2.0f32 * cp_w_star - cost_oracle;
        loss[i] = max_term + lin_term;
    }
    loss
}

/// Evaluates causal potential outcomes on CPU: mu0(x) + T * tau(x).
pub fn causal_forward_cpu(
    x: &[f32],
    treatment_t: &[f32],
    w_tau: &[f32],
    b_tau: &[f32],
    w_mu0: &[f32],
    b_mu0: &[f32],
    batch: usize,
    in_dim: usize,
) -> Vec<f32> {
    let tau = linear_forward_cpu(x, w_tau, b_tau, batch, in_dim, 1);
    let mu0 = linear_forward_cpu(x, w_mu0, b_mu0, batch, in_dim, 1);
    let mut y_factual = vec![0.0f32; batch];
    for i in 0..batch {
        y_factual[i] = mu0[i] + treatment_t[i] * tau[i];
    }
    y_factual
}

/// Evaluates Equation Learner (EQL) symbolic forward on CPU.
pub fn symbolic_forward_cpu(x: &[f32], w_eql: &[f32], b_eql: &[f32], batch: usize) -> Vec<f32> {
    let in_dim = 4;
    let mut eql_features = vec![0.0f32; batch * 4];
    for i in 0..batch {
        let x_i = &x[i * in_dim..(i + 1) * in_dim];
        eql_features[i * 4 + 0] = x_i[0];
        eql_features[i * 4 + 1] = x_i[1].sin();
        eql_features[i * 4 + 2] = x_i[2].cos();
        eql_features[i * 4 + 3] = x_i[0] * x_i[3];
    }
    linear_forward_cpu(&eql_features, w_eql, b_eql, batch, 4, 1)
}

#[cfg(feature = "wgpu")]
pub fn linear_forward_wgpu(
    ctx: &WgpuContext,
    x: &[f32],
    w: &[f32],
    b: &[f32],
    batch: usize,
    in_dim: usize,
    out_dim: usize,
) -> Option<Vec<f32>> {
    let dims = Dims {
        batch: batch as u32,
        in_dim: in_dim as u32,
        out_dim: out_dim as u32,
    };

    let wg_x = (batch as u32).div_ceil(16);
    let wg_y = (out_dim as u32).div_ceil(16);

    let bytes = crate::wgpu_util::run_compute_shader(
        ctx,
        WGSL_SHADER,
        "main",
        &[
            bytemuck::cast_slice(x),
            bytemuck::cast_slice(w),
            bytemuck::cast_slice(b),
        ],
        batch * out_dim * std::mem::size_of::<f32>(),
        Some(bytemuck::bytes_of(&dims)),
        (wg_x, wg_y, 1),
    )?;

    Some(bytemuck::cast_slice::<u8, f32>(&bytes).to_vec())
}

#[cfg(not(feature = "wgpu"))]
pub fn linear_forward_wgpu(
    _ctx: &WgpuContext,
    _x: &[f32],
    _w: &[f32],
    _b: &[f32],
    _batch: usize,
    _in_dim: usize,
    _out_dim: usize,
) -> Option<Vec<f32>> {
    None
}

pub fn linear_forward(
    device: &KernelDevice,
    x: &[f32],
    w: &[f32],
    b: &[f32],
    batch: usize,
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    match device {
        KernelDevice::Wgpu(ctx) => {
            if let Some(res) = linear_forward_wgpu(ctx, x, w, b, batch, in_dim, out_dim) {
                return res;
            }
            linear_forward_cpu(x, w, b, batch, in_dim, out_dim)
        }
        KernelDevice::Cpu => linear_forward_cpu(x, w, b, batch, in_dim, out_dim),
    }
}
