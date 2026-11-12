//! Native Candle implementation for Category 04 Topic 02:
//! 02_modern_sign_and_factored_optimisers in 04_training_methods_and_optimisers/01_gradient_and_curvature_optimisers.

use candle_core::{Result, Tensor};

/// Linear anchor module for Category 04 Topic 02
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

/// Lion: EvoLved Sign Momentum (Chen et al. 2023)
/// update = sign(beta1 * exp_avg + (1 - beta1) * grad)
/// param = param * (1 - lr * weight_decay) - lr * update
/// exp_avg = beta2 * exp_avg + (1 - beta2) * grad
pub fn step_lion(
    param: &Tensor,
    grad: &Tensor,
    exp_avg: &Tensor,
    lr: f64,
    beta1: f64,
    beta2: f64,
    weight_decay: f64,
) -> Result<(Tensor, Tensor)> {
    let m_term = (exp_avg * beta1)?;
    let g_term = (grad * (1.0 - beta1))?;
    let interp = (&m_term + &g_term)?;

    // Signum operator:
    // Tensor::sign() is elementwise sign (-1.0, 0.0, 1.0)
    let update = interp.sign()?;

    let shrunk_p = (param * (1.0 - lr * weight_decay))?;
    let step = (&update * lr)?;
    let new_p = (&shrunk_p - &step)?;

    let new_m_term = (exp_avg * beta2)?;
    let new_g_term = (grad * (1.0 - beta2))?;
    let new_exp_avg = (&new_m_term + &new_g_term)?;

    Ok((new_p, new_exp_avg))
}

/// Adafactor 2D Factored Second Moment Update (Shazeer & Stern 2018)
pub fn step_adafactor_2d(
    param: &Tensor,
    grad: &Tensor,
    r_buf: &Tensor,
    c_buf: &Tensor,
    step: u32,
    lr: f64,
    beta2: f64,
    eps1: f64,
    eps2: f64,
    clip_threshold: f64,
    weight_decay: f64,
) -> Result<(Tensor, Tensor, Tensor)> {
    let decay_rate = 1.0 - (step as f64).powf(-beta2);
    let s = (grad.sqr()?.affine(1.0, eps1))?;

    let mean_s_row = s.mean(1)?;
    let mean_s_col = s.mean(0)?;

    let r_term = (r_buf * decay_rate)?;
    let new_r_term = (&mean_s_row * (1.0 - decay_rate))?;
    let new_r = (&r_term + &new_r_term)?;

    let c_term = (c_buf * decay_rate)?;
    let new_c_term = (&mean_s_col * (1.0 - decay_rate))?;
    let new_c = (&c_term + &new_c_term)?;

    let r_mean = new_r.mean_all()?.to_scalar::<f32>()? as f64;

    // v_approx = (new_r.unsqueeze(1) * new_c.unsqueeze(0)) / (r_mean + eps1)
    let r_col = new_r.unsqueeze(1)?;
    let c_row = new_c.unsqueeze(0)?;
    let outer = r_col.matmul(&c_row)?;
    let v_approx = (&outer * (1.0 / (r_mean + eps1)))?;

    let denom = (v_approx.sqrt()?.affine(1.0, eps2))?;
    let u = grad.div(&denom)?;

    let u_sq = u.sqr()?;
    let mean_u_sq = u_sq.mean_all()?.to_scalar::<f32>()? as f64;
    let rms_u = mean_u_sq.sqrt();
    let raw_clip = rms_u / clip_threshold;
    let clip_factor = if raw_clip > 1.0 { raw_clip } else { 1.0 };
    let u_clipped = (&u * (1.0 / clip_factor))?;

    let shrunk_p = (param * (1.0 - lr * weight_decay))?;
    let step_p = (&u_clipped * lr)?;
    let new_p = (&shrunk_p - &step_p)?;

    Ok((new_p, new_r, new_c))
}

/// AdEMAMix Dual Momentum Mixture Update (Pagliardini et al. 2024)
pub fn step_ademamix(
    param: &Tensor,
    grad: &Tensor,
    m1_buf: &Tensor,
    m2_buf: &Tensor,
    v_buf: &Tensor,
    step: u32,
    lr: f64,
    beta1: f64,
    beta2: f64,
    beta3: f64,
    alpha: f64,
    eps: f64,
    weight_decay: f64,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let new_m1 = ((m1_buf * beta1)? + (grad * (1.0 - beta1))?)?;
    let new_m2 = ((m2_buf * beta3)? + (grad * (1.0 - beta3))?)?;
    let new_v = ((v_buf * beta2)? + (grad.sqr()? * (1.0 - beta2))?)?;

    let bc1 = 1.0 - beta1.powi(step as i32);
    let bc2 = 1.0 - beta2.powi(step as i32);
    let bc3 = 1.0 - beta3.powi(step as i32);

    let m1_hat = (&new_m1 * (1.0 / bc1))?;
    let m2_hat = (&new_m2 * (1.0 / bc3))?;
    let v_hat = (&new_v * (1.0 / bc2))?;

    let combined_m = (&m1_hat + &(&m2_hat * alpha)?)?;
    let denom = (v_hat.sqrt()?.affine(1.0, eps))?;
    let u = combined_m.div(&denom)?;

    let shrunk_p = (param * (1.0 - lr * weight_decay))?;
    let step_p = (&u * lr)?;
    let new_p = (&shrunk_p - &step_p)?;

    Ok((new_p, new_m1, new_m2, new_v))
}
