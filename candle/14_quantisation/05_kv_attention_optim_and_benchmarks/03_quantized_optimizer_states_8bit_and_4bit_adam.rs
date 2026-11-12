//! Quantized Optimizer States: 8-bit and 4-bit Adam
//! in candle/14_quantisation/05_kv_attention_optim_and_benchmarks.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Block-wise symmetric dynamic INT8 quantization for first momentum m_t.
pub fn quantize_8bit_first_moment(
    m: &Tensor,
    block_size: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let vec = m.flatten_all()?.to_vec1::<f32>()?;
    let mut deq_vec = Vec::with_capacity(vec.len());
    let mut int_vec = Vec::with_capacity(vec.len());
    let mut scales_vec = Vec::new();

    for chunk in vec.chunks(block_size) {
        let abs_max = chunk
            .iter()
            .fold(0.0f32, |acc, &v| acc.max(v.abs()))
            .max(1e-7);
        let scale = (abs_max / 127.0f32).max(1e-7);
        scales_vec.push(scale);
        let inv_scale = 1.0f32 / scale;

        for &v in chunk {
            let q = (v * inv_scale).round().clamp(-127.0f32, 127.0f32);
            int_vec.push(q);
            deq_vec.push(q * scale);
        }
    }

    let n_scales = scales_vec.len();
    let deq = Tensor::from_vec(deq_vec, m.shape(), m.device())?;
    let int_t = Tensor::from_vec(int_vec, m.shape(), m.device())?;
    let scales = Tensor::from_vec(scales_vec, &[n_scales], m.device())?;
    Ok((deq, int_t, scales))
}

/// Block-wise non-linear log2 quantization for non-negative second momentum v_t.
pub fn quantize_8bit_second_moment(
    v: &Tensor,
    block_size: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let vec = v.flatten_all()?.to_vec1::<f32>()?;
    let mut deq_vec = Vec::with_capacity(vec.len());
    let mut int_vec = Vec::with_capacity(vec.len());
    let mut max_vec = Vec::new();

    for chunk in vec.chunks(block_size) {
        let max_val = chunk
            .iter()
            .fold(0.0f32, |acc, &val| acc.max(val))
            .max(1e-8);
        max_vec.push(max_val);
        let inv_max = 1.0f32 / max_val;

        for &val in chunk {
            let norm = (val * inv_max).max(1e-8);
            let log_v = norm.log2();
            let q = ((log_v + 24.0) * (255.0 / 24.0))
                .round()
                .clamp(0.0f32, 255.0f32);
            int_vec.push(q);
            let deq_norm = 2.0f32.powf((q * (24.0 / 255.0)) - 24.0);
            deq_vec.push(deq_norm * max_val);
        }
    }

    let n_max = max_vec.len();
    let deq = Tensor::from_vec(deq_vec, v.shape(), v.device())?;
    let int_t = Tensor::from_vec(int_vec, v.shape(), v.device())?;
    let max_t = Tensor::from_vec(max_vec, &[n_max], v.device())?;
    Ok((deq, int_t, max_t))
}

/// 4-bit block-wise optimizer state quantization: signed 4-bit m_t and log 4-bit v_t.
pub fn quantize_4bit_moments(
    m: &Tensor,
    v: &Tensor,
    block_size: usize,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let m_vec = m.flatten_all()?.to_vec1::<f32>()?;
    let mut m_deq_vec = Vec::with_capacity(m_vec.len());
    let mut m_int_vec = Vec::with_capacity(m_vec.len());

    for chunk in m_vec.chunks(block_size) {
        let abs_max = chunk
            .iter()
            .fold(0.0f32, |acc, &val| acc.max(val.abs()))
            .max(1e-7);
        let scale = (abs_max / 7.0f32).max(1e-7);
        let inv_scale = 1.0f32 / scale;
        for &val in chunk {
            let q = (val * inv_scale).round().clamp(-7.0f32, 7.0f32);
            m_int_vec.push(q);
            m_deq_vec.push(q * scale);
        }
    }

    let v_vec = v.flatten_all()?.to_vec1::<f32>()?;
    let mut v_deq_vec = Vec::with_capacity(v_vec.len());
    let mut v_int_vec = Vec::with_capacity(v_vec.len());

    for chunk in v_vec.chunks(block_size) {
        let max_val = chunk
            .iter()
            .fold(0.0f32, |acc, &val| acc.max(val))
            .max(1e-8);
        let inv_max = 1.0f32 / max_val;
        for &val in chunk {
            let norm = (val * inv_max).max(1e-6);
            let log_v = norm.log2();
            let q = ((log_v + 15.0) * (15.0 / 15.0))
                .round()
                .clamp(0.0f32, 15.0f32);
            v_int_vec.push(q);
            let deq_norm = 2.0f32.powf((q * (15.0 / 15.0)) - 15.0);
            v_deq_vec.push(deq_norm * max_val);
        }
    }

    let m_deq = Tensor::from_vec(m_deq_vec, m.shape(), m.device())?;
    let m_int = Tensor::from_vec(m_int_vec, m.shape(), m.device())?;
    let v_deq = Tensor::from_vec(v_deq_vec, v.shape(), v.device())?;
    let v_int = Tensor::from_vec(v_int_vec, v.shape(), v.device())?;
    Ok((m_deq, m_int, v_deq, v_int))
}

/// Full AdamW parameter update with explicit momentum integration.
pub fn adamw_update_step(
    theta: &Tensor,
    grad: &Tensor,
    m_prev_deq: &Tensor,
    v_prev_deq: &Tensor,
    beta1: f64,
    beta2: f64,
    lr: f64,
    weight_decay: f64,
    step: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let m_t = ((m_prev_deq * beta1)? + (grad * (1.0 - beta1))?)?;
    let grad_sq = grad.sqr()?;
    let v_t = ((v_prev_deq * beta2)? + (&grad_sq * (1.0 - beta2))?)?;

    let bc1 = 1.0 - beta1.powi(step as i32);
    let bc2 = 1.0 - beta2.powi(step as i32);
    let m_hat = (&m_t / bc1)?;
    let v_hat = (&v_t / bc2)?;

    let v_hat_sqrt = v_hat.sqrt()?;
    let denom = v_hat_sqrt.affine(1.0, 1e-8)?;
    let adam_step = m_hat.broadcast_div(&denom)?;

    let wd_step = (theta * weight_decay)?;
    let update = (adam_step + wd_step)?;
    let theta_next = (theta - (update * lr)?)?;

    Ok((theta_next, m_t, v_t))
}
