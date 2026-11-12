//! Native Candle implementation for Category 04 Topic 03:
//! 03_large_batch_and_layerwise_optimisers in 04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers.

use candle_core::{Result, Tensor};

/// Linear anchor module for Category 04 Topic 03
#[derive(Debug, Clone)]
pub struct LinearAnchor {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl LinearAnchor {
    pub fn new(weight: Tensor, bias: Tensor) -> Self {
        Self { weight, bias }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_w = x.matmul(&self.weight)?;
        x_w.broadcast_add(&self.bias)
    }
}

/// LARS: Layer-wise Adaptive Rate Scaling (You et al. 2017)
pub fn step_lars(
    param: &Tensor,
    grad: &Tensor,
    v_buf: &Tensor,
    lr: f64,
    momentum: f64,
    weight_decay: f64,
    trust_coef: f64,
    eps: f64,
) -> Result<(Tensor, Tensor, Tensor)> {
    let w_sq = param.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
    let g_sq = grad.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
    let w_norm = w_sq.sqrt();
    let g_norm = g_sq.sqrt();

    let trust_ratio = if w_norm > 0.0 && g_norm > 0.0 {
        trust_coef * w_norm / (g_norm + weight_decay * w_norm + eps)
    } else {
        1.0
    };

    let local_lr = lr * trust_ratio;
    let wd_p = (param * weight_decay)?;
    let total_g = (grad + &wd_p)?;
    let scaled_g = (total_g * local_lr)?;

    let mom_v = (v_buf * momentum)?;
    let new_v = (&mom_v + &scaled_g)?;
    let new_p = (param - &new_v)?;

    let ratio_tensor = Tensor::new(&[trust_ratio as f32], param.device())?;
    Ok((new_p, new_v, ratio_tensor))
}

/// LAMB: Layer-wise Adaptive Moments for Batch training (You et al. 2019)
pub fn step_lamb(
    param: &Tensor,
    grad: &Tensor,
    m_buf: &Tensor,
    v_buf: &Tensor,
    step: u32,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let new_m = ((m_buf * beta1)? + (grad * (1.0 - beta1))?)?;
    let new_v = ((v_buf * beta2)? + (grad.sqr()? * (1.0 - beta2))?)?;

    let bc1 = 1.0 - beta1.powi(step as i32);
    let bc2 = 1.0 - beta2.powi(step as i32);

    let m_hat = (&new_m * (1.0 / bc1))?;
    let v_hat = (&new_v * (1.0 / bc2))?;

    let denom = (v_hat.sqrt()?.affine(1.0, eps))?;
    let adam_step = m_hat.div(&denom)?;

    let wd_p = (param * weight_decay)?;
    let r = (&adam_step + &wd_p)?;

    let w_sq = param.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
    let r_sq = r.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
    let w_norm = w_sq.sqrt();
    let r_norm = r_sq.sqrt();

    let trust_ratio = if w_norm > 0.0 && r_norm > 0.0 {
        w_norm / r_norm
    } else {
        1.0
    };

    let step_r = (&r * (lr * trust_ratio))?;
    let new_p = (param - &step_r)?;

    let ratio_tensor = Tensor::new(&[trust_ratio as f32], param.device())?;
    Ok((new_p, new_m, new_v, ratio_tensor))
}

/// Gradient Noise Scale (McCandlish et al. 2018)
pub fn compute_gradient_noise_scale(per_sample_grads: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
    // per_sample_grads: [B, D]
    let (b_size, _dim) = per_sample_grads.dims2()?;
    let g_mean = per_sample_grads.mean(0)?;

    let g_norm_sq = g_mean.sqr()?.sum_all()?.reshape(&[1])?;
    let diff = per_sample_grads.broadcast_sub(&g_mean)?;
    let sum_sq_diff = diff.sqr()?.sum_all()?;
    let trace_sigma = (&sum_sq_diff * (1.0 / ((b_size - 1) as f64)))?.reshape(&[1])?;

    let denom = (g_norm_sq.affine(1.0, 1e-12))?;
    let b_simple = trace_sigma.div(&denom)?;

    Ok((g_norm_sq, trace_sigma, b_simple))
}
