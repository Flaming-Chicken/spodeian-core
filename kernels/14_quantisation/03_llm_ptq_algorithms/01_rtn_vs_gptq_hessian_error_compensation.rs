//! CPU and WGSL kernels for RTN vs GPTQ Hessian Error Compensation.

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

/// WGSL shader for GPTQ rank-1 trailing weight matrix error compensation update:
/// W[r, j] -= (err[j] / H_inv_cc) * H_inv[r, c]
pub const GPTQ_UPDATE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> weight: array<f32>;
@group(0) @binding(1) var<storage, read> h_inv_col: array<f32>;
@group(0) @binding(2) var<storage, read> err_row: array<f32>;

struct Uniforms {
    rows: u32,
    cols: u32,
    h_inv_cc: f32,
};
@group(0) @binding(3) var<uniform> params: Uniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let r = global_id.x;
    let c = global_id.y;
    if (r >= params.rows || c >= params.cols) {
        return;
    }
    let idx = r * params.cols + c;
    let step = err_row[c] / params.h_inv_cc;
    let delta = step * h_inv_col[r];
    weight[idx] = weight[idx] - delta;
}
"#;

/// CPU reference implementation of GPTQ rank-1 trailing weight update.
pub fn gptq_trailing_update_cpu(
    weight_trailing: &mut [f32],
    h_inv_col: &[f32],
    err_row: &[f32],
    h_inv_cc: f32,
    rows: usize,
    cols: usize,
) {
    if h_inv_cc == 0.0 {
        return;
    }
    for r in 0..rows {
        let h_rc = h_inv_col[r];
        for c in 0..cols {
            let step = err_row[c] / h_inv_cc;
            let delta = step * h_rc;
            weight_trailing[r * cols + c] -= delta;
        }
    }
}

/// Auto dispatcher: executes CPU reference fallback.
pub fn gptq_trailing_update_auto(
    weight_trailing: &mut [f32],
    h_inv_col: &[f32],
    err_row: &[f32],
    h_inv_cc: f32,
    rows: usize,
    cols: usize,
) {
    gptq_trailing_update_cpu(weight_trailing, h_inv_col, err_row, h_inv_cc, rows, cols);
}
