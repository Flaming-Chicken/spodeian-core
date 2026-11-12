//! Native Candle implementation for Category 04 Topic 01:
//! 01_first_order_and_adaptive_optimisers in 04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers.

use candle_core::{Result, Tensor};

/// Linear anchor module for Category 04 Topic 01
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
        let x_w = x.matmul(&self.weight.t()?)?;
        x_w.broadcast_add(&self.bias)
    }
}

/// SGD with classical momentum and L2 weight decay:
/// g = grad + weight_decay * param
/// v_{t+1} = momentum * v_t + g
/// p_{t+1} = param - lr * v_{t+1}
pub fn step_sgd_momentum(
    param: &Tensor,
    grad: &Tensor,
    momentum_buf: &Tensor,
    lr: f64,
    momentum: f64,
    weight_decay: f64,
) -> Result<(Tensor, Tensor)> {
    let wd_p = (param * weight_decay)?;
    let g = (grad + &wd_p)?;
    let mom_v = (momentum_buf * momentum)?;
    let new_v = (&mom_v + &g)?;
    let step = (&new_v * lr)?;
    let new_p = (param - &step)?;
    Ok((new_p, new_v))
}

/// Adagrad with cumulative squared gradient accumulator:
/// g = grad + weight_decay * param
/// G_{t+1} = G_t + g^2
/// p_{t+1} = param - lr * g / (sqrt(G_{t+1}) + eps)
pub fn step_adagrad(
    param: &Tensor,
    grad: &Tensor,
    accum_buf: &Tensor,
    lr: f64,
    eps: f64,
    weight_decay: f64,
) -> Result<(Tensor, Tensor)> {
    let wd_p = (param * weight_decay)?;
    let g = (grad + &wd_p)?;
    let g_sq = g.sqr()?;
    let new_accum = (accum_buf + &g_sq)?;
    let denom = (new_accum.sqrt()?.affine(1.0, eps))?;
    let update = g.div(&denom)?;
    let step = (update * lr)?;
    let new_p = (param - &step)?;
    Ok((new_p, new_accum))
}

/// RMSprop with exponential moving average of squared gradients:
/// g = grad + weight_decay * param
/// v_{t+1} = alpha * v_t + (1 - alpha) * g^2
/// p_{t+1} = param - lr * g / (sqrt(v_{t+1}) + eps)
pub fn step_rmsprop(
    param: &Tensor,
    grad: &Tensor,
    square_buf: &Tensor,
    lr: f64,
    alpha: f64,
    eps: f64,
    weight_decay: f64,
) -> Result<(Tensor, Tensor)> {
    let wd_p = (param * weight_decay)?;
    let g = (grad + &wd_p)?;
    let g_sq = g.sqr()?;
    let v_term = (square_buf * alpha)?;
    let g_term = (g_sq * (1.0 - alpha))?;
    let new_v = (&v_term + &g_term)?;
    let denom = (new_v.sqrt()?.affine(1.0, eps))?;
    let update = g.div(&denom)?;
    let step = (update * lr)?;
    let new_p = (param - &step)?;
    Ok((new_p, new_v))
}

/// Adam with coupled L2 regularization:
/// g = grad + weight_decay * param
/// m_{t+1} = beta1 * m_t + (1 - beta1) * g
/// v_{t+1} = beta2 * v_t + (1 - beta2) * g^2
/// m_hat = m_{t+1} / (1 - beta1^t)
/// v_hat = v_{t+1} / (1 - beta2^t)
/// p_{t+1} = param - lr * m_hat / (sqrt(v_hat) + eps)
pub fn step_adam(
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
) -> Result<(Tensor, Tensor, Tensor)> {
    let wd_p = (param * weight_decay)?;
    let g = (grad + &wd_p)?;
    let g_sq = g.sqr()?;

    let new_m = ((m_buf * beta1)? + (g * (1.0 - beta1))?)?;
    let new_v = ((v_buf * beta2)? + (g_sq * (1.0 - beta2))?)?;

    let bias_corr1 = 1.0 - beta1.powi(step as i32);
    let bias_corr2 = 1.0 - beta2.powi(step as i32);

    let m_hat = (&new_m * (1.0 / bias_corr1))?;
    let v_hat = (&new_v * (1.0 / bias_corr2))?;

    let denom = (v_hat.sqrt()?.affine(1.0, eps))?;
    let update = m_hat.div(&denom)?;
    let step_p = (update * lr)?;
    let new_p = (param - &step_p)?;
    Ok((new_p, new_m, new_v))
}

/// Decoupled AdamW (Loshchilov & Hutter 2017):
/// m_{t+1} = beta1 * m_t + (1 - beta1) * grad (pure gradient!)
/// v_{t+1} = beta2 * v_t + (1 - beta2) * grad^2
/// m_hat = m_{t+1} / (1 - beta1^t)
/// v_hat = v_{t+1} / (1 - beta2^t)
/// p_{t+1} = param * (1 - lr * weight_decay) - lr * m_hat / (sqrt(v_hat) + eps)
pub fn step_adamw(
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
) -> Result<(Tensor, Tensor, Tensor)> {
    let g_sq = grad.sqr()?;

    let new_m = ((m_buf * beta1)? + (grad * (1.0 - beta1))?)?;
    let new_v = ((v_buf * beta2)? + (g_sq * (1.0 - beta2))?)?;

    let bias_corr1 = 1.0 - beta1.powi(step as i32);
    let bias_corr2 = 1.0 - beta2.powi(step as i32);

    let m_hat = (&new_m * (1.0 / bias_corr1))?;
    let v_hat = (&new_v * (1.0 / bias_corr2))?;

    let denom = (v_hat.sqrt()?.affine(1.0, eps))?;
    let update = m_hat.div(&denom)?;
    let step_p = (update * lr)?;

    let shrunk_p = (param * (1.0 - lr * weight_decay))?;
    let new_p = (&shrunk_p - &step_p)?;
    Ok((new_p, new_m, new_v))
}

/// AdaBelief (Zhuang et al. 2020):
/// Adapts step-size according to the belief in gradient direction:
/// m_{t+1} = beta1 * m_t + (1 - beta1) * grad
/// diff = grad - m_{t+1}
/// s_{t+1} = beta2 * s_t + (1 - beta2) * diff^2 + eps
/// m_hat = m_{t+1} / (1 - beta1^t)
/// s_hat = s_{t+1} / (1 - beta2^t)
/// p_{t+1} = param * (1 - lr * weight_decay) - lr * m_hat / (sqrt(s_hat) + eps)
pub fn step_adabelief(
    param: &Tensor,
    grad: &Tensor,
    m_buf: &Tensor,
    s_buf: &Tensor,
    step: u32,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
) -> Result<(Tensor, Tensor, Tensor)> {
    let new_m = ((m_buf * beta1)? + (grad * (1.0 - beta1))?)?;
    let diff = (grad - &new_m)?;
    let diff_sq = diff.sqr()?;

    let s_term = (s_buf * beta2)?;
    let diff_term = (diff_sq * (1.0 - beta2))?;
    let new_s = (&s_term + &diff_term)?.affine(1.0, eps)?;

    let bias_corr1 = 1.0 - beta1.powi(step as i32);
    let bias_corr2 = 1.0 - beta2.powi(step as i32);

    let m_hat = (&new_m * (1.0 / bias_corr1))?;
    let s_hat = (&new_s * (1.0 / bias_corr2))?;

    let denom = (s_hat.sqrt()?.affine(1.0, eps))?;
    let update = m_hat.div(&denom)?;
    let step_p = (update * lr)?;

    let shrunk_p = (param * (1.0 - lr * weight_decay))?;
    let new_p = (&shrunk_p - &step_p)?;
    Ok((new_p, new_m, new_s))
}
