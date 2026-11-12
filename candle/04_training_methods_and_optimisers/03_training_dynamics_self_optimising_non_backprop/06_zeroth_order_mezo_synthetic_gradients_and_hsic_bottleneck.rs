//! Candle Implementation for Category 04 Topic 20 (Subfolder 03 Topic 06):
//! Zeroth-Order Optimization: MeZO, Synthetic Gradients (DNI), and HSIC Bottleneck.
//!
//! Provides zero-Python, memory-safe implementations of:
//! 1. MeZO directional derivative and parameter update (Malladi et al., 2023).
//! 2. Decoupled Neural Interfaces (DNI) synthetic gradient prediction and MSE loss (Jaderberg et al., 2017).
//! 3. Hilbert-Schmidt Independence Criterion (HSIC) with linear kernels (Ma et al., 2020).
//! 4. Linear anchor evaluation for transitive multi-track parity.

use candle_core::{Device, Result, Tensor};
use std::collections::BTreeMap;
use std::path::Path;

/// Linear Anchor module for cross-track parity verification.
pub struct LinearAnchor {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl LinearAnchor {
    pub fn new(weight: Tensor, bias: Tensor) -> Self {
        Self { weight, bias }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let xw = x.matmul(&self.weight)?;
        xw.broadcast_add(&self.bias)
    }
}

/// MeZO directional derivative and parameter update step (Malladi et al., 2023).
///
/// loss(p) = 0.5 * sum((p - theta_star)^2)
/// dir_deriv = (loss(theta + eps * z) - loss(theta - eps * z)) / (2 * eps)
/// theta_next = theta - eta * dir_deriv * z
pub fn mezo_step(
    theta: &Tensor,
    z: &Tensor,
    theta_star: &Tensor,
    eps: f64,
    eta: f64,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let dev = theta.device();
    let eps_t = Tensor::new(eps as f32, dev)?;
    let eps_z = z.broadcast_mul(&eps_t)?;

    let theta_pos = theta.add(&eps_z)?;
    let theta_neg = theta.sub(&eps_z)?;

    let diff_pos = theta_pos.sub(theta_star)?;
    let diff_neg = theta_neg.sub(theta_star)?;

    let half = Tensor::new(0.5f32, dev)?;
    let loss_pos = (diff_pos.sqr()?.sum_all()? * &half)?;
    let loss_neg = (diff_neg.sqr()?.sum_all()? * &half)?;

    let two_eps = Tensor::new((2.0 * eps) as f32, dev)?;
    let dir_deriv = (loss_pos.sub(&loss_neg)? / two_eps)?;

    let eta_t = Tensor::new(eta as f32, dev)?;
    let step = z.broadcast_mul(&dir_deriv)?.broadcast_mul(&eta_t)?;
    let theta_next = theta.sub(&step)?;

    // Reshape scalars to [1] for exact parity matching
    let loss_pos_1 = loss_pos.reshape(1)?;
    let loss_neg_1 = loss_neg.reshape(1)?;
    let dir_deriv_1 = dir_deriv.reshape(1)?;

    Ok((loss_pos_1, loss_neg_1, dir_deriv_1, theta_next))
}

/// Decoupled Neural Interfaces (DNI) synthetic gradient prediction and MSE loss (Jaderberg et al., 2017).
///
/// pred_grad = h @ w_synth + b_synth
/// dni_loss = 0.5 * mean((pred_grad - target_grad)^2)
pub fn dni_synthetic_gradient(
    h: &Tensor,
    w_synth: &Tensor,
    b_synth: &Tensor,
    target_grad: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let dev = h.device();
    let pred_grad = h.matmul(w_synth)?.broadcast_add(b_synth)?;
    let diff = pred_grad.sub(target_grad)?;

    let half = Tensor::new(0.5f32, dev)?;
    let dni_loss = (diff.sqr()?.mean_all()? * half)?.reshape(1)?;

    Ok((pred_grad, dni_loss))
}

/// Hilbert-Schmidt Independence Criterion (HSIC) with linear kernels (Ma et al., 2020).
///
/// K = H H^T, L = Y Y^T
/// H_c = I - 1/B * 1 1^T
/// K_c = H_c K H_c, L_c = H_c L H_c
/// HSIC = 1 / (B - 1)^2 * Tr(K_c L_c)
pub fn hsic_bottleneck(h: &Tensor, y: &Tensor) -> Result<Tensor> {
    let dev = h.device();
    let b = h.dim(0)?;
    let b_f = b as f32;

    let k_mat = h.matmul(&h.t()?)?;
    let l_mat = y.matmul(&y.t()?)?;

    // Centering matrix H_c = I - 1/B * ones
    let eye = Tensor::eye(b, candle_core::DType::F32, dev)?;
    let ones = (Tensor::ones((b, b), candle_core::DType::F32, dev)? / (b_f as f64))?;
    let h_c = eye.sub(&ones)?;

    let k_c = h_c.matmul(&k_mat)?.matmul(&h_c)?;
    let l_c = h_c.matmul(&l_mat)?.matmul(&h_c)?;

    // For symmetric matrices K_c and L_c, Tr(K_c L_c) = sum(K_c * L_c)
    let tr = k_c.mul(&l_c)?.sum_all()?;

    let denom = (b_f - 1.0) * (b_f - 1.0);
    let scale = Tensor::new(1.0f32 / denom, dev)?;
    let score = (tr * scale)?.reshape(1)?;

    Ok(score)
}

/// Unified Candle container for Category 04 Topic 20 verification.
pub struct ZerothOrderUnifiedModule {
    pub theta: Tensor,
    pub z: Tensor,
    pub theta_star: Tensor,
    pub golden_loss_pos: Tensor,
    pub golden_loss_neg: Tensor,
    pub golden_dir_deriv: Tensor,
    pub golden_theta_next: Tensor,

    pub h: Tensor,
    pub w_synth: Tensor,
    pub b_synth: Tensor,
    pub target_grad: Tensor,
    pub golden_pred_grad: Tensor,
    pub golden_dni_loss: Tensor,

    pub hsic_h: Tensor,
    pub hsic_y: Tensor,
    pub golden_hsic_score: Tensor,

    pub anchor: LinearAnchor,
    pub anchor_x: Tensor,
    pub golden_anchor_out: Tensor,
}

impl ZerothOrderUnifiedModule {
    pub fn new(weights_path: &Path) -> Result<Self> {
        let dev = &Device::Cpu;
        let tensors = candle_core::safetensors::load(weights_path, dev)?;

        let theta = tensors.get("theta").unwrap().clone();
        let z = tensors.get("z").unwrap().clone();
        let theta_star = tensors.get("theta_star").unwrap().clone();
        let golden_loss_pos = tensors.get("golden_loss_pos").unwrap().clone();
        let golden_loss_neg = tensors.get("golden_loss_neg").unwrap().clone();
        let golden_dir_deriv = tensors.get("golden_dir_deriv").unwrap().clone();
        let golden_theta_next = tensors.get("golden_theta_next").unwrap().clone();

        let h = tensors.get("h").unwrap().clone();
        let w_synth = tensors.get("w_synth").unwrap().clone();
        let b_synth = tensors.get("b_synth").unwrap().clone();
        let target_grad = tensors.get("target_grad").unwrap().clone();
        let golden_pred_grad = tensors.get("golden_pred_grad").unwrap().clone();
        let golden_dni_loss = tensors.get("golden_dni_loss").unwrap().clone();

        let hsic_h = tensors.get("hsic_h").unwrap().clone();
        let hsic_y = tensors.get("hsic_y").unwrap().clone();
        let golden_hsic_score = tensors.get("golden_hsic_score").unwrap().clone();

        let anchor_w = tensors.get("anchor_w").unwrap().clone();
        let anchor_b = tensors.get("anchor_b").unwrap().clone();
        let anchor_x = tensors.get("anchor_x").unwrap().clone();
        let golden_anchor_out = tensors.get("golden_anchor_out").unwrap().clone();

        let anchor = LinearAnchor::new(anchor_w, anchor_b);

        Ok(Self {
            theta,
            z,
            theta_star,
            golden_loss_pos,
            golden_loss_neg,
            golden_dir_deriv,
            golden_theta_next,
            h,
            w_synth,
            b_synth,
            target_grad,
            golden_pred_grad,
            golden_dni_loss,
            hsic_h,
            hsic_y,
            golden_hsic_score,
            anchor,
            anchor_x,
            golden_anchor_out,
        })
    }

    pub fn verify_invariants(&self, atol: f64) -> Result<BTreeMap<String, f64>> {
        let mut errs = BTreeMap::new();

        // 1. MeZO invariant
        let (lp, ln, dd, th_next) = mezo_step(&self.theta, &self.z, &self.theta_star, 1e-3, 0.05)?;
        let err_lp = (&lp - &self.golden_loss_pos)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        let err_ln = (&ln - &self.golden_loss_neg)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        let err_dd = (&dd - &self.golden_dir_deriv)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        let err_th = (&th_next - &self.golden_theta_next)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        let err_mezo = err_lp.max(err_ln).max(err_dd).max(err_th);
        assert!(
            err_mezo < atol,
            "MeZO invariant failed: {err_mezo:.2e} >= {atol:.2e}"
        );
        errs.insert("mezo_err".to_string(), err_mezo);

        // 2. DNI invariant
        let (pred_g, dni_l) =
            dni_synthetic_gradient(&self.h, &self.w_synth, &self.b_synth, &self.target_grad)?;
        let err_pg = (&pred_g - &self.golden_pred_grad)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        let err_dl = (&dni_l - &self.golden_dni_loss)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        let err_dni = err_pg.max(err_dl);
        assert!(
            err_dni < atol,
            "DNI invariant failed: {err_dni:.2e} >= {atol:.2e}"
        );
        errs.insert("dni_err".to_string(), err_dni);

        // 3. HSIC invariant
        let hsic_s = hsic_bottleneck(&self.hsic_h, &self.hsic_y)?;
        let err_hsic = (&hsic_s - &self.golden_hsic_score)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        assert!(
            err_hsic < atol,
            "HSIC invariant failed: {err_hsic:.2e} >= {atol:.2e}"
        );
        errs.insert("hsic_err".to_string(), err_hsic);

        // 4. Linear Anchor invariant
        let anchor_out = self.anchor.forward(&self.anchor_x)?;
        let err_anchor = (&anchor_out - &self.golden_anchor_out)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()? as f64;
        assert!(
            err_anchor < atol,
            "Anchor invariant failed: {err_anchor:.2e} >= {atol:.2e}"
        );
        errs.insert("anchor_err".to_string(), err_anchor);

        Ok(errs)
    }
}
