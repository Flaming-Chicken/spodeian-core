//! Pipeline Parallelism GPipe, 1F1B, Zero-Bubble, and DualPipe in candle/16_distributed_and_scalable.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Calculates standard GPipe / 1F1B bubble fraction: (P - 1) / (M + P - 1).
pub fn compute_pipeline_bubble_fraction(num_stages: usize, num_microbatches: usize) -> f64 {
    (num_stages - 1) as f64 / (num_microbatches + num_stages - 1) as f64
}

/// 2-stage pipeline forward evaluation across microbatches.
pub fn pipeline_forward_step(
    x_mb: &Tensor,
    w0: &Tensor,
    b0: &Tensor,
    w1: &Tensor,
    b1: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let m = x_mb.dim(0)?;
    let mut h_list = Vec::with_capacity(m);
    let mut y_list = Vec::with_capacity(m);

    for i in 0..m {
        let x_i = x_mb.get(i)?;
        let h_i = x_i.matmul(w0)?.broadcast_add(b0)?;
        let y_i = h_i.matmul(w1)?.broadcast_add(b1)?;
        h_list.push(h_i);
        y_list.push(y_i);
    }

    let h_refs: Vec<&Tensor> = h_list.iter().collect();
    let y_refs: Vec<&Tensor> = y_list.iter().collect();
    let h_mb = Tensor::stack(&h_refs, 0)?;
    let y_mb = Tensor::stack(&y_refs, 0)?;
    Ok((h_mb, y_mb))
}

/// DualPipe / Zero-Bubble split backward computation: B_A (activation) and B_W (weight).
pub fn dualpipe_backward_step(
    x_mb: &Tensor,
    h_mb: &Tensor,
    grad_y_mb: &Tensor,
    w0: &Tensor,
    w1: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    let m = x_mb.dim(0)?;
    let w0_t = w0.t()?;
    let w1_t = w1.t()?;

    let mut grad_w0 = Tensor::zeros(w0.shape(), w0.dtype(), w0.device())?;
    let mut grad_w1 = Tensor::zeros(w1.shape(), w1.dtype(), w1.device())?;
    let mut grad_x_list = Vec::with_capacity(m);

    for i in 0..m {
        let g_y_i = grad_y_mb.get(i)?;
        let h_i = h_mb.get(i)?;
        let x_i = x_mb.get(i)?;

        // Stage 1
        let b_a_1 = g_y_i.matmul(&w1_t)?;
        let b_w_1 = h_i.t()?.matmul(&g_y_i)?;
        grad_w1 = (grad_w1 + b_w_1)?;

        // Stage 0
        let b_a_0 = b_a_1.matmul(&w0_t)?;
        let b_w_0 = x_i.t()?.matmul(&b_a_1)?;
        grad_w0 = (grad_w0 + b_w_0)?;

        grad_x_list.push(b_a_0);
    }

    let grad_x_refs: Vec<&Tensor> = grad_x_list.iter().collect();
    let grad_x_mb = Tensor::stack(&grad_x_refs, 0)?;
    Ok((grad_w0, grad_w1, grad_x_mb))
}
