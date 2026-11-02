//! PEFT LoRA Rslora Loraplus and DoRA
//! in candle/15_compression_peft_and_speculative_inference/03_parameter_efficient_finetuning_peft.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Standard LoRA forward pass: h = x W0 + (alpha / r) * (x A) B.
pub fn lora_forward(
    x: &Tensor,
    w0: &Tensor,
    a: &Tensor,
    b: &Tensor,
    alpha: f64,
    r: usize,
) -> Result<Tensor> {
    let scale = alpha / (r as f64);
    let base_out = x.matmul(w0)?;
    let adapter_out = (x.matmul(a)?.matmul(b)? * scale)?;
    base_out.broadcast_add(&adapter_out)
}

/// Rank-Stabilized LoRA (rsLoRA) forward pass: h = x W0 + (alpha / sqrt(r)) * (x A) B.
pub fn rslora_forward(
    x: &Tensor,
    w0: &Tensor,
    a: &Tensor,
    b: &Tensor,
    alpha: f64,
    r: usize,
) -> Result<Tensor> {
    let scale = alpha / ((r as f64).sqrt());
    let base_out = x.matmul(w0)?;
    let adapter_out = (x.matmul(a)?.matmul(b)? * scale)?;
    base_out.broadcast_add(&adapter_out)
}

/// Weight-Decomposed Low-Rank Adaptation (DoRA) forward pass.
pub fn dora_forward(
    x: &Tensor,
    w0: &Tensor,
    a: &Tensor,
    b: &Tensor,
    m: &Tensor,
    alpha: f64,
    r: usize,
) -> Result<Tensor> {
    let (d_in, d_out) = w0.dims2()?;
    let scale = alpha / (r as f64);
    let delta = (a.matmul(b)? * scale)?;
    let v = (w0.broadcast_add(&delta))?;

    let v_vec = v.flatten_all()?.to_vec1::<f32>()?;
    let m_vec = m.flatten_all()?.to_vec1::<f32>()?;

    let mut w_dora_vec = vec![0.0f32; d_in * d_out];
    for c in 0..d_out {
        let mut col_sum_sq = 0.0f32;
        for i in 0..d_in {
            let val = v_vec[i * d_out + c];
            col_sum_sq += val * val;
        }
        let col_norm = col_sum_sq.sqrt().max(1e-8);
        let m_c = m_vec[c];
        for i in 0..d_in {
            w_dora_vec[i * d_out + c] = (m_c / col_norm) * v_vec[i * d_out + c];
        }
    }

    let w_dora = Tensor::from_vec(w_dora_vec, (d_in, d_out), w0.device())?;
    x.matmul(&w_dora)
}
