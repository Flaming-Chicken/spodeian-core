//! Memory Optimization Selective Checkpointing Offloading and Grad Hooks
//! in candle/12_compilation_and_performance.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Selective activation checkpointing with recomputed SiLU gate backward.
pub fn selective_checkpoint_silu_gate(
    x: &Tensor,
    w: &Tensor,
    b: &Tensor,
    gate: &Tensor,
    grad_out: &Tensor,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
    let x_vec = x.to_vec2::<f32>()?;
    let w_vec = w.to_vec2::<f32>()?;
    let b_vec = b.to_vec1::<f32>()?;
    let gate_vec = gate.to_vec2::<f32>()?;
    let g_out_vec = grad_out.to_vec2::<f32>()?;

    let batch = x_vec.len();
    let d_in = w_vec.len();
    let d_out = w_vec[0].len();

    let mut y = vec![vec![0.0f32; d_out]; batch];
    let mut gz = vec![vec![0.0f32; d_out]; batch];
    let mut grad_gate = vec![vec![0.0f32; d_out]; batch];

    for i in 0..batch {
        for j in 0..d_out {
            let mut z = b_vec[j];
            for k in 0..d_in {
                z += x_vec[i][k] * w_vec[k][j];
            }
            let sig_z = 1.0f32 / (1.0f32 + (-z).exp());
            let a = z * sig_z;
            y[i][j] = a * gate_vec[i][j];

            let g_a = g_out_vec[i][j] * gate_vec[i][j];
            let da_dz = sig_z * (1.0f32 + z * (1.0f32 - sig_z));
            gz[i][j] = g_a * da_dz;
            grad_gate[i][j] = g_out_vec[i][j] * a;
        }
    }

    // grad_w = x^T @ gz
    let mut grad_w = vec![vec![0.0f32; d_out]; d_in];
    for k in 0..d_in {
        for j in 0..d_out {
            let mut sum = 0.0f32;
            for i in 0..batch {
                sum += x_vec[i][k] * gz[i][j];
            }
            grad_w[k][j] = sum;
        }
    }

    // grad_b = sum_i gz[i]
    let mut grad_b = vec![0.0f32; d_out];
    for j in 0..d_out {
        let mut sum = 0.0f32;
        for i in 0..batch {
            sum += gz[i][j];
        }
        grad_b[j] = sum;
    }

    // grad_x = gz @ w^T
    let mut grad_x = vec![vec![0.0f32; d_in]; batch];
    for i in 0..batch {
        for k in 0..d_in {
            let mut sum = 0.0f32;
            for j in 0..d_out {
                sum += gz[i][j] * w_vec[k][j];
            }
            grad_x[i][k] = sum;
        }
    }

    let dev = x.device();
    Ok((
        Tensor::new(y, dev)?,
        Tensor::new(grad_w, dev)?,
        Tensor::new(grad_b, dev)?,
        Tensor::new(grad_x, dev)?,
        Tensor::new(grad_gate, dev)?,
    ))
}

/// Post-accumulate gradient hook optimizer step simulation.
pub fn post_accumulate_grad_hook_step(
    w: &Tensor,
    grad_w: &Tensor,
    running_m: &Tensor,
    beta: f32,
    lr: f32,
) -> Result<(Tensor, Tensor)> {
    let m_new = (running_m.affine(beta as f64, 0.0)? + grad_w.affine((1.0 - beta) as f64, 0.0)?)?;
    let w_new = (w - m_new.affine(lr as f64, 0.0)?)?;
    Ok((w_new, m_new))
}
