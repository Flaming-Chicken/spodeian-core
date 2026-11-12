//! Hardware Kernels (WGSL + Native CPU fallback) for Category 04 Topic 10:
//! 03_coordinate_alternating_projections_and_majorization_minimization in 04_training_methods_and_optimisers/02_constrained_pareto_and_blackbox_optimisation.

use crate::context::KernelDevice;

/// WGSL Compute Shader for Concave-Convex Procedure (CCCP) majorization step: (2 * x) / (1 + x^2).
pub const CCCP_STEP_SHADER: &str = r#"
struct CccpUniforms {
    len: u32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> uniforms: CccpUniforms;
@group(0) @binding(1) var<storage, read> param_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> param_out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.len) {
        return;
    }
    let x = param_in[idx];
    param_out[idx] = (2.0 * x) / (1.0 + x * x);
}
"#;

/// WGSL Compute Shader for Block Coordinate Descent (BCD) step.
pub const BCD_STEP_SHADER: &str = r#"
struct BcdUniforms {
    len: u32,
    coupling: f32,
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0) var<uniform> uniforms: BcdUniforms;
@group(0) @binding(1) var<storage, read> param_in: array<f32>;
@group(0) @binding(2) var<storage, read> b_in: array<f32>;
@group(0) @binding(3) var<storage, read> a_diag_in: array<f32>;
@group(0) @binding(4) var<storage, read_write> param_out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.len) {
        return;
    }
    let p = param_in[idx];
    let b = b_in[idx];
    let a = a_diag_in[idx];
    param_out[idx] = (b - uniforms.coupling * p) / a;
}
"#;

/// Project onto halfspace <normal, x> <= offset.
pub fn proj_halfspace_cpu(x: &[f32], normal: &[f32], offset: f32) -> Vec<f32> {
    let d = x.len();
    let mut dot = 0.0f64;
    let mut norm_sq = 0.0f64;
    for i in 0..d {
        dot += (x[i] as f64) * (normal[i] as f64);
        norm_sq += (normal[i] as f64) * (normal[i] as f64);
    }
    if dot <= (offset as f64) {
        return x.to_vec();
    }
    let factor = (dot - (offset as f64)) / norm_sq;
    let mut out = vec![0.0f32; d];
    for i in 0..d {
        out[i] = ((x[i] as f64) - factor * (normal[i] as f64)) as f32;
    }
    out
}

/// Project onto box [lower, upper]^d.
pub fn proj_box_cpu(x: &[f32], lower: f32, upper: f32) -> Vec<f32> {
    x.iter().map(|&v| v.clamp(lower, upper)).collect()
}

/// CPU implementation of Dykstra's Alternating Projections step.
pub fn step_dykstra_cpu(
    y_prev: &[f32],
    p_prev: &[f32],
    q_prev: &[f32],
    normal: &[f32],
    offset: f32,
    lower: f32,
    upper: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let d = y_prev.len();

    // Step 1: y_prev + p_prev
    let mut yp = vec![0.0f32; d];
    for i in 0..d {
        yp[i] = y_prev[i] + p_prev[i];
    }
    let x_curr = proj_halfspace_cpu(&yp, normal, offset);

    // p_curr = y_prev + p_prev - x_curr
    let mut p_curr = vec![0.0f32; d];
    for i in 0..d {
        p_curr[i] = yp[i] - x_curr[i];
    }

    // Step 2: x_curr + q_prev
    let mut xq = vec![0.0f32; d];
    for i in 0..d {
        xq[i] = x_curr[i] + q_prev[i];
    }
    let y_curr = proj_box_cpu(&xq, lower, upper);

    // q_curr = x_curr + q_prev - y_curr
    let mut q_curr = vec![0.0f32; d];
    for i in 0..d {
        q_curr[i] = xq[i] - y_curr[i];
    }

    (x_curr, y_curr, p_curr, q_curr)
}

/// CPU implementation of Block Coordinate Descent (BCD) step.
pub fn step_bcd_cpu(param: &[f32], b: &[f32], a_diag: &[f32], coupling: f32) -> Vec<f32> {
    let d = param.len();
    let mut out = vec![0.0f32; d];
    for i in 0..d {
        out[i] = (b[i] - coupling * param[i]) / a_diag[i];
    }
    out
}

/// CPU implementation of Iteratively Reweighted Least Squares (IRLS) step.
pub fn step_irls_cpu(param: &[f32], target: &[f32], mu: f32, eps: f32) -> Vec<f32> {
    let d = param.len();
    let mut out = vec![0.0f32; d];
    for i in 0..d {
        let r = ((param[i] as f64) - (target[i] as f64)).abs();
        let w = 1.0 / (r + (eps as f64));
        let tgt = target[i] as f64;
        let p = param[i] as f64;
        let m = mu as f64;
        out[i] = ((w * tgt + m * p) / (w + m)) as f32;
    }
    out
}

/// CPU implementation of Concave-Convex Procedure (CCCP) majorization step.
pub fn step_cccp_cpu(param: &[f32]) -> Vec<f32> {
    let d = param.len();
    let mut out = vec![0.0f32; d];
    for i in 0..d {
        let x = param[i] as f64;
        out[i] = ((2.0 * x) / (1.0 + x * x)) as f32;
    }
    out
}

/// Linear anchor forward pass: x @ w + b.
pub fn anchor_forward_cpu(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * out_dim];
    for r in 0..rows {
        for c in 0..out_dim {
            let mut sum = b[c];
            for k in 0..in_dim {
                sum += x[r * in_dim + k] * w[k * out_dim + c];
            }
            out[r * out_dim + c] = sum;
        }
    }
    out
}

/// Auto-dispatchers with CPU fallback.
pub fn step_dykstra_auto(
    _device: &KernelDevice,
    y_prev: &[f32],
    p_prev: &[f32],
    q_prev: &[f32],
    normal: &[f32],
    offset: f32,
    lower: f32,
    upper: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    step_dykstra_cpu(y_prev, p_prev, q_prev, normal, offset, lower, upper)
}

pub fn step_bcd_auto(
    _device: &KernelDevice,
    param: &[f32],
    b: &[f32],
    a_diag: &[f32],
    coupling: f32,
) -> Vec<f32> {
    step_bcd_cpu(param, b, a_diag, coupling)
}

pub fn step_irls_auto(
    _device: &KernelDevice,
    param: &[f32],
    target: &[f32],
    mu: f32,
    eps: f32,
) -> Vec<f32> {
    step_irls_cpu(param, target, mu, eps)
}

pub fn step_cccp_auto(_device: &KernelDevice, param: &[f32]) -> Vec<f32> {
    step_cccp_cpu(param)
}
