//! PEFT SVD and Adaptive PiSSA Adalora Vera
//! in candle/15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// PiSSA forward pass: h = x W_res + (x B) A.
pub fn pissa_forward(x: &Tensor, w_res: &Tensor, b: &Tensor, a: &Tensor) -> Result<Tensor> {
    let base_out = x.matmul(w_res)?;
    let adapter_out = x.matmul(b)?.matmul(a)?;
    base_out.broadcast_add(&adapter_out)
}

/// AdaLoRA forward pass with dynamic rank mask: h = x W0 + x Delta W.
pub fn adalora_forward(
    x: &Tensor,
    w0: &Tensor,
    p: &Tensor,
    lambda_diag: &Tensor,
    q: &Tensor,
    mask: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let effective_lambda = (lambda_diag * mask)?;
    let r = effective_lambda.dims1()?;
    let l_vec = effective_lambda.to_vec1::<f32>()?;

    let mut diag_vec = vec![0.0f32; r * r];
    for i in 0..r {
        diag_vec[i * r + i] = l_vec[i];
    }
    let diag_mat = Tensor::from_vec(diag_vec, (r, r), lambda_diag.device())?;

    let delta_w = p.matmul(&diag_mat)?.matmul(q)?;
    let base_out = x.matmul(w0)?;
    let adapted_out = x.matmul(&delta_w)?;
    let h = base_out.broadcast_add(&adapted_out)?;

    Ok((h, delta_w))
}

/// VeRA (Vector-based Random Matrix Adaptation) forward pass.
pub fn vera_forward(
    x: &Tensor,
    w0: &Tensor,
    a_shared: &Tensor,
    b_shared: &Tensor,
    d_vec: &Tensor,
    b_vec: &Tensor,
) -> Result<Tensor> {
    let base_out = x.matmul(w0)?;
    let x_a = x.matmul(a_shared)?;
    let scaled_xa = x_a.broadcast_mul(d_vec)?;
    let xa_b = scaled_xa.matmul(b_shared)?;
    let adapter_out = xa_b.broadcast_mul(b_vec)?;
    base_out.broadcast_add(&adapter_out)
}
