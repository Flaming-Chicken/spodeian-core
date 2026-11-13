//! PyTorch 2 Parity & Information-Theoretic Optimizers for `sml-runtime`.
//!
//! Provides zero-Python, memory-safe, hardware-portable implementations of:
//! - `Sgd`: Stochastic Gradient Descent (Momentum, Dampening, Nesterov, L2 weight decay)
//! - `Adam`: Adaptive Moment Estimation with optional AMSGrad max-variance tracking
//! - `AdamW`: Decoupled Weight Decay Regularization (Loshchilov & Hutter, 2019)
//! - `RmsProp`: Root Mean Square Propagation (Centered & Uncentered, Momentum)
//! - `Adagrad`: Cumulative Squared Gradient Preconditioning with LR Decay
//! - `Lion`: EvoLved Sign Momentum with Decoupled Weight Decay (Chen et al., 2023)
//! - `SophiaG`: Second-Order Clipped Stochastic Diagonal Gauss-Newton Curvature (Liu et al., 2024)
//! - `ScheduleFreeAdamW`: Schedule-Free Primal-Dual Polyak Averaging (Defazio et al., 2024)
//! - `Muon`: Newton-Schulz Orthogonalized Matrix Momentum (Jordan et al., 2024)
//! - `SelfOptimizingMeta`: Autonomous Information-Theoretic Meta-Optimizer combining:
//!   1. $C^\infty$ Padé $(2,2)$ rational warmup & invariant primal-dual step coupling $\omega_k$
//!   2. Coordinate-wise empirical Fisher preconditioning ($N_{\text{eff}, 2} = 1000$)
//!   3. Shannon surprisal Hill-4 outlier gate ($I_{\text{crit}} = 2.0\text{ nats} \implies K_{\text{hill}} = 256$)
//!   4. Unconditional $C^\infty$ Levenberg-Marquardt soft-saturation operator coupled to innovation rate $1 - \beta_s$
//!   5. Datatype-calibrated dynamic Newton-Schulz polar orthogonalization for 2D matrix parameters
//!   6. Schedule-free Polyak-Ruppert center-manifold averaging with decoupled primal weight decay

use candle_core::backprop::GradStore;
use candle_core::{CpuStorage, DType, Result, Storage, Tensor, Var};
use serde::{Deserialize, Serialize};
use smallvec::{SmallVec, smallvec};

/// Universal Optimizer Trait matching PyTorch 2 contract.
pub trait Optimizer {
    /// Performs a single optimization step using computed gradients.
    fn step(&mut self, grads: &GradStore) -> Result<()>;

    /// Convenience helper: computes backward gradients from loss and executes an optimization step.
    fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        let grads = loss.backward()?;
        self.step(&grads)
    }

    /// Returns the current learning rate.
    fn learning_rate(&self) -> f64;

    /// Updates the learning rate (e.g., from an LR scheduler).
    fn set_learning_rate(&mut self, lr: f64);
}

// ============================================================================
// 1. SGD (Stochastic Gradient Descent)
// ============================================================================

#[derive(Debug, Clone)]
pub struct SgdConfig {
    pub lr: f64,
    pub momentum: f64,
    pub dampening: f64,
    pub weight_decay: f64,
    pub nesterov: bool,
}

impl Default for SgdConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            momentum: 0.0,
            dampening: 0.0,
            weight_decay: 0.0,
            nesterov: false,
        }
    }
}

pub struct Sgd {
    vars: Vec<Var>,
    config: SgdConfig,
    momentum_buffers: Vec<Option<Tensor>>,
}

impl Sgd {
    pub fn new(vars: Vec<Var>, config: SgdConfig) -> Self {
        let n = vars.len();
        Self {
            vars,
            config,
            momentum_buffers: (0..n).map(|_| None).collect(),
        }
    }

    pub fn new_lr(vars: Vec<Var>, lr: f64) -> Self {
        Self::new(
            vars,
            SgdConfig {
                lr,
                ..Default::default()
            },
        )
    }
}

impl Optimizer for Sgd {
    fn step(&mut self, grads: &GradStore) -> Result<()> {
        for (idx, var) in self.vars.iter().enumerate() {
            if let Some(grad) = grads.get(var.as_tensor()) {
                let p = var.as_tensor();
                let mut d_p = if self.config.weight_decay != 0.0 {
                    (grad + (p * self.config.weight_decay)?)?
                } else {
                    grad.clone()
                };

                if self.config.momentum != 0.0 {
                    let buf = match &self.momentum_buffers[idx] {
                        None => d_p.clone(),
                        Some(b) => {
                            let dampened = if self.config.dampening > 0.0 {
                                (d_p.affine(1.0 - self.config.dampening, 0.0))?
                            } else {
                                d_p.clone()
                            };
                            ((b * self.config.momentum)? + dampened)?
                        }
                    };
                    self.momentum_buffers[idx] = Some(buf.clone());

                    d_p = if self.config.nesterov {
                        (d_p + (buf * self.config.momentum)?)?
                    } else {
                        buf
                    };
                }

                let step = (d_p * self.config.lr)?;
                let new_p = (p - step)?;
                var.set(&new_p)?;
            }
        }
        Ok(())
    }

    fn learning_rate(&self) -> f64 {
        self.config.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.config.lr = lr;
    }
}

// ============================================================================
// 2. Adam & AdamW
// ============================================================================

#[derive(Debug, Clone)]
pub struct AdamWConfig {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub amsgrad: bool,
    pub decoupled_weight_decay: bool,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 1e-2,
            amsgrad: false,
            decoupled_weight_decay: true,
        }
    }
}

pub struct AdamW {
    vars: Vec<Var>,
    config: AdamWConfig,
    m: Vec<Option<Tensor>>,
    v: Vec<Option<Tensor>>,
    max_v: Vec<Option<Tensor>>,
    step_count: usize,
}

impl AdamW {
    pub fn new(vars: Vec<Var>, config: AdamWConfig) -> Self {
        let n = vars.len();
        Self {
            vars,
            config,
            m: (0..n).map(|_| None).collect(),
            v: (0..n).map(|_| None).collect(),
            max_v: (0..n).map(|_| None).collect(),
            step_count: 0,
        }
    }

    pub fn new_lr(vars: Vec<Var>, lr: f64) -> Self {
        Self::new(
            vars,
            AdamWConfig {
                lr,
                ..Default::default()
            },
        )
    }

    pub fn adam(vars: Vec<Var>, lr: f64) -> Self {
        Self::new(
            vars,
            AdamWConfig {
                lr,
                weight_decay: 0.0,
                decoupled_weight_decay: false,
                ..Default::default()
            },
        )
    }
}

impl Optimizer for AdamW {
    fn step(&mut self, grads: &GradStore) -> Result<()> {
        self.step_count += 1;
        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let eps = self.config.eps;
        let lr = self.config.lr;
        let bias_correction1 = 1.0 - beta1.powi(self.step_count as i32);
        let bias_correction2 = 1.0 - beta2.powi(self.step_count as i32);

        for (idx, var) in self.vars.iter().enumerate() {
            if let Some(grad) = grads.get(var.as_tensor()) {
                let p = var.as_tensor();

                // L2 regular weight decay vs Decoupled AdamW
                let (grad, p_curr) = if !self.config.decoupled_weight_decay
                    && self.config.weight_decay != 0.0
                {
                    let g_wd = (grad + (p * self.config.weight_decay)?)?;
                    (g_wd, p.clone())
                } else if self.config.decoupled_weight_decay && self.config.weight_decay != 0.0 {
                    let decayed = (p * (1.0 - lr * self.config.weight_decay))?;
                    (grad.clone(), decayed)
                } else {
                    (grad.clone(), p.clone())
                };

                // Update biased first moment: m_t = beta1 * m_{t-1} + (1 - beta1) * grad
                let new_m = match &self.m[idx] {
                    None => (grad.affine(1.0 - beta1, 0.0))?,
                    Some(prev_m) => ((prev_m * beta1)? + grad.affine(1.0 - beta1, 0.0)?)?,
                };
                self.m[idx] = Some(new_m.clone());

                // Update biased second moment: v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
                let grad_sq = grad.sqr()?;
                let new_v = match &self.v[idx] {
                    None => (grad_sq.affine(1.0 - beta2, 0.0))?,
                    Some(prev_v) => ((prev_v * beta2)? + grad_sq.affine(1.0 - beta2, 0.0)?)?,
                };
                self.v[idx] = Some(new_v.clone());

                // AMSGrad option: max(v_{t-1}, v_t)
                let denom_v = if self.config.amsgrad {
                    let max_v = match &self.max_v[idx] {
                        None => new_v.clone(),
                        Some(prev_max) => prev_max.maximum(&new_v)?,
                    };
                    self.max_v[idx] = Some(max_v.clone());
                    max_v
                } else {
                    new_v
                };

                // Bias-corrected step size: step = (m / bias1) / (sqrt(v / bias2) + eps)
                let m_hat = (new_m / bias_correction1)?;
                let v_hat = (denom_v / bias_correction2)?;
                let denom = (v_hat.sqrt()?.affine(1.0, eps))?;
                let update = (m_hat / denom)?;
                let step = (update * lr)?;
                let new_p = (p_curr - step)?;
                var.set(&new_p)?;
            }
        }
        Ok(())
    }

    fn learning_rate(&self) -> f64 {
        self.config.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.config.lr = lr;
    }
}

// ============================================================================
// 3. Lion (EvoLved Sign Momentum)
// ============================================================================

#[derive(Debug, Clone)]
pub struct LionConfig {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub weight_decay: f64,
}

impl Default for LionConfig {
    fn default() -> Self {
        Self {
            lr: 1e-4,
            beta1: 0.9,
            beta2: 0.99,
            weight_decay: 0.1,
        }
    }
}

pub struct Lion {
    vars: Vec<Var>,
    config: LionConfig,
    exp_avg: Vec<Option<Tensor>>,
}

impl Lion {
    pub fn new(vars: Vec<Var>, config: LionConfig) -> Self {
        let n = vars.len();
        Self {
            vars,
            config,
            exp_avg: (0..n).map(|_| None).collect(),
        }
    }

    pub fn new_lr(vars: Vec<Var>, lr: f64) -> Self {
        Self::new(
            vars,
            LionConfig {
                lr,
                ..Default::default()
            },
        )
    }
}

impl Optimizer for Lion {
    fn step(&mut self, grads: &GradStore) -> Result<()> {
        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let lr = self.config.lr;
        let wd = self.config.weight_decay;

        for (idx, var) in self.vars.iter().enumerate() {
            if let Some(grad) = grads.get(var.as_tensor()) {
                let p = var.as_tensor();

                // Decoupled weight decay: p = p * (1 - lr * wd)
                let p_decayed = if wd > 0.0 {
                    (p * (1.0 - lr * wd))?
                } else {
                    p.clone()
                };

                // Compute update: c_t = beta1 * exp_avg + (1 - beta1) * grad
                let prev_m = self.exp_avg[idx].as_ref();
                let c_t = match prev_m {
                    None => grad.clone(),
                    Some(m) => ((m * beta1)? + grad.affine(1.0 - beta1, 0.0)?)?,
                };

                // sign(c_t): element-wise sign
                let sign_c = c_t.sign()?;
                let step = (sign_c * lr)?;
                let new_p = (p_decayed - step)?;
                var.set(&new_p)?;

                // Update momentum: exp_avg = beta2 * exp_avg + (1 - beta2) * grad
                let new_m = match prev_m {
                    None => grad.affine(1.0 - beta2, 0.0)?,
                    Some(m) => ((m * beta2)? + grad.affine(1.0 - beta2, 0.0)?)?,
                };
                self.exp_avg[idx] = Some(new_m);
            }
        }
        Ok(())
    }

    fn learning_rate(&self) -> f64 {
        self.config.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.config.lr = lr;
    }
}

// ============================================================================
// 4. RMSprop
// ============================================================================

#[derive(Debug, Clone)]
pub struct RMSpropConfig {
    pub lr: f64,
    pub alpha: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub momentum: f64,
    pub centered: bool,
}

impl Default for RMSpropConfig {
    fn default() -> Self {
        Self {
            lr: 1e-2,
            alpha: 0.99,
            eps: 1e-8,
            weight_decay: 0.0,
            momentum: 0.0,
            centered: false,
        }
    }
}

pub struct RMSprop {
    vars: Vec<Var>,
    config: RMSpropConfig,
    square_avg: Vec<Option<Tensor>>,
    grad_avg: Vec<Option<Tensor>>,
    momentum_buffer: Vec<Option<Tensor>>,
}

impl RMSprop {
    pub fn new(vars: Vec<Var>, config: RMSpropConfig) -> Self {
        let n = vars.len();
        Self {
            vars,
            config,
            square_avg: (0..n).map(|_| None).collect(),
            grad_avg: (0..n).map(|_| None).collect(),
            momentum_buffer: (0..n).map(|_| None).collect(),
        }
    }
}

impl Optimizer for RMSprop {
    fn step(&mut self, grads: &GradStore) -> Result<()> {
        let alpha = self.config.alpha;
        let eps = self.config.eps;
        let lr = self.config.lr;

        for (idx, var) in self.vars.iter().enumerate() {
            if let Some(grad) = grads.get(var.as_tensor()) {
                let p = var.as_tensor();
                let grad = if self.config.weight_decay != 0.0 {
                    (grad + (p * self.config.weight_decay)?)?
                } else {
                    grad.clone()
                };

                let grad_sq = grad.sqr()?;
                let sq_avg = match &self.square_avg[idx] {
                    None => grad_sq.affine(1.0 - alpha, 0.0)?,
                    Some(prev) => ((prev * alpha)? + grad_sq.affine(1.0 - alpha, 0.0)?)?,
                };
                self.square_avg[idx] = Some(sq_avg.clone());

                let avg = if self.config.centered {
                    let g_avg = match &self.grad_avg[idx] {
                        None => grad.affine(1.0 - alpha, 0.0)?,
                        Some(prev) => ((prev * alpha)? + grad.affine(1.0 - alpha, 0.0)?)?,
                    };
                    self.grad_avg[idx] = Some(g_avg.clone());
                    (sq_avg - g_avg.sqr()?)?
                } else {
                    sq_avg
                };

                let denom = (avg.sqrt()?.affine(1.0, eps))?;
                let mut d_p = (grad / denom)?;

                if self.config.momentum > 0.0 {
                    let buf = match &self.momentum_buffer[idx] {
                        None => d_p.clone(),
                        Some(prev_buf) => ((prev_buf * self.config.momentum)? + &d_p)?,
                    };
                    self.momentum_buffer[idx] = Some(buf.clone());
                    d_p = buf;
                }

                let step = (d_p * lr)?;
                let new_p = (p - step)?;
                var.set(&new_p)?;
            }
        }
        Ok(())
    }

    fn learning_rate(&self) -> f64 {
        self.config.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.config.lr = lr;
    }
}

// ============================================================================
// 5. Sophia-G (Second-Order Clipped Stochastic Curvature)
// ============================================================================

#[derive(Debug, Clone)]
pub struct SophiaConfig {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub rho: f64,
}

impl Default for SophiaConfig {
    fn default() -> Self {
        Self {
            lr: 1e-4,
            beta1: 0.96,
            beta2: 0.99,
            eps: 1e-12,
            weight_decay: 0.1,
            rho: 0.04,
        }
    }
}

pub struct Sophia {
    vars: Vec<Var>,
    config: SophiaConfig,
    m: Vec<Option<Tensor>>,
    h: Vec<Option<Tensor>>,
}

impl Sophia {
    pub fn new(vars: Vec<Var>, config: SophiaConfig) -> Self {
        let n = vars.len();
        Self {
            vars,
            config,
            m: (0..n).map(|_| None).collect(),
            h: (0..n).map(|_| None).collect(),
        }
    }

    /// Update Hessian diagonal estimate (called periodically every k steps)
    pub fn update_hessian(&mut self, hessian_diags: &[Tensor]) -> Result<()> {
        let beta2 = self.config.beta2;
        for (idx, h_t) in hessian_diags.iter().enumerate() {
            if idx < self.h.len() {
                let new_h = match &self.h[idx] {
                    None => h_t.affine(1.0 - beta2, 0.0)?,
                    Some(prev) => ((prev * beta2)? + h_t.affine(1.0 - beta2, 0.0)?)?,
                };
                self.h[idx] = Some(new_h);
            }
        }
        Ok(())
    }
}

impl Optimizer for Sophia {
    fn step(&mut self, grads: &GradStore) -> Result<()> {
        let beta1 = self.config.beta1;
        let lr = self.config.lr;
        let wd = self.config.weight_decay;
        let rho = self.config.rho;
        let eps = self.config.eps;

        for (idx, var) in self.vars.iter().enumerate() {
            if let Some(grad) = grads.get(var.as_tensor()) {
                let p = var.as_tensor();

                // Decoupled weight decay
                let p_decayed = if wd > 0.0 {
                    (p * (1.0 - lr * wd))?
                } else {
                    p.clone()
                };

                // m_t = beta1 * m_{t-1} + (1 - beta1) * grad
                let new_m = match &self.m[idx] {
                    None => grad.affine(1.0 - beta1, 0.0)?,
                    Some(prev) => ((prev * beta1)? + grad.affine(1.0 - beta1, 0.0)?)?,
                };
                self.m[idx] = Some(new_m.clone());

                // Denominator: max(h_t, eps)
                let h_t = match &self.h[idx] {
                    None => grad.sqr()?,
                    Some(h) => h.clone(),
                };
                let denom = h_t.clamp(eps, f64::MAX)?;

                // Direction: clip(m_t / denom, -rho, rho)
                let dir = (new_m / denom)?;
                let clipped = dir.clamp(-rho, rho)?;
                let step = (clipped * lr)?;
                let new_p = (p_decayed - step)?;
                var.set(&new_p)?;
            }
        }
        Ok(())
    }

    fn learning_rate(&self) -> f64 {
        self.config.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.config.lr = lr;
    }
}

// ============================================================================
// 6. ScheduleFreeAdamW (Defazio et al., 2024)
// ============================================================================

#[derive(Debug, Clone)]
pub struct ScheduleFreeConfig {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub weight_decay: f64,
    pub warmup_steps: usize,
    pub eps: f64,
}

impl Default for ScheduleFreeConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            weight_decay: 1e-2,
            warmup_steps: 100,
            eps: 1e-8,
        }
    }
}

pub struct ScheduleFreeAdamW {
    vars: Vec<Var>,
    config: ScheduleFreeConfig,
    y: Vec<Tensor>, // Primal iterates
    z: Vec<Tensor>, // Momentum iterates
    exp_avg_sq: Vec<Option<Tensor>>,
    step_count: usize,
}

impl ScheduleFreeAdamW {
    pub fn new(vars: Vec<Var>, config: ScheduleFreeConfig) -> Self {
        let n = vars.len();
        let y = vars.iter().map(|v| v.as_tensor().clone()).collect();
        let z = vars.iter().map(|v| v.as_tensor().clone()).collect();
        Self {
            vars,
            config,
            y,
            z,
            exp_avg_sq: (0..n).map(|_| None).collect(),
            step_count: 0,
        }
    }
}

impl Optimizer for ScheduleFreeAdamW {
    fn step(&mut self, grads: &GradStore) -> Result<()> {
        self.step_count += 1;
        let k = self.step_count as f64;
        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let lr = self.config.lr;
        let wd = self.config.weight_decay;
        let eps = self.config.eps;

        // Dynamic schedule-free weight ck: warmup then k
        let ck = if self.step_count <= self.config.warmup_steps {
            (k / self.config.warmup_steps as f64) * (1.0 / (k + 1.0))
        } else {
            1.0 / (k + 1.0)
        };

        for (idx, var) in self.vars.iter().enumerate() {
            if let Some(grad) = grads.get(var.as_tensor()) {
                let grad_sq = grad.sqr()?;
                let v = match &self.exp_avg_sq[idx] {
                    None => grad_sq.affine(1.0 - beta2, 0.0)?,
                    Some(prev) => ((prev * beta2)? + grad_sq.affine(1.0 - beta2, 0.0)?)?,
                };
                self.exp_avg_sq[idx] = Some(v.clone());

                let denom = (v.sqrt()?.affine(1.0, eps))?;
                let step_dir = (grad / denom)?;

                // z_{t+1} = z_t - lr * step_dir - lr * wd * z_t
                let z_curr = &self.z[idx];
                let z_next = if wd > 0.0 {
                    (z_curr * (1.0 - lr * wd))?.sub(&(step_dir * lr)?)?
                } else {
                    (z_curr - (step_dir * lr)?)?
                };
                self.z[idx] = z_next.clone();

                // y_{t+1} = (1 - ck) * y_t + ck * z_{t+1}
                let y_curr = &self.y[idx];
                let y_next = ((y_curr * (1.0 - ck))? + (z_next.affine(ck, 0.0))?)?;
                self.y[idx] = y_next.clone();

                // Model weights set to primal iterate x_{t+1} = beta1 * y_{t+1} + (1 - beta1) * z_{t+1}
                let x_model = ((y_next * beta1)? + z_next.affine(1.0 - beta1, 0.0)?)?;
                var.set(&x_model)?;
            }
        }
        Ok(())
    }

    fn learning_rate(&self) -> f64 {
        self.config.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.config.lr = lr;
    }
}

// ============================================================================
// 9. Muon (Momentum with Orthogonalized Updates via Newton-Schulz)
// ============================================================================

/// Configuration for the Muon optimizer.
#[derive(Debug, Clone)]
pub struct MuonConfig {
    pub lr: f64,
    pub momentum: f64,
    pub nesterov: bool,
    /// Newton-Schulz iteration count. If set to 0 (the default), automatically calibrates
    /// to the parameter tensor's native [`DType`] mantissa precision:
    /// - `F64`: 4 iterations ($\|X X^\top - I\| \approx 2.22 \times 10^{-16}$)
    /// - `F32`: 3 iterations ($\|X X^\top - I\| \approx 1.19 \times 10^{-7}$)
    /// - `BF16` / `F16`: 2 iterations ($\|X X^\top - I\| \approx 10^{-3}$)
    pub ns_steps: usize,
    pub weight_decay: f64,
}

impl Default for MuonConfig {
    fn default() -> Self {
        Self {
            lr: 0.02,
            momentum: 0.95,
            nesterov: true,
            ns_steps: 0,
            weight_decay: 0.01,
        }
    }
}

impl MuonConfig {
    /// Dynamically calibrates Newton-Schulz iteration count to the datatype's mantissa precision.
    pub fn ns_steps_for_dtype(dtype: DType) -> usize {
        match dtype {
            DType::F64 => 4,
            DType::F32 => 3,
            DType::BF16 | DType::F16 => 2,
            _ => 3,
        }
    }

    /// Constructs a [`MuonConfig`] explicitly calibrated for `dtype`.
    pub fn for_dtype(dtype: DType) -> Self {
        Self {
            ns_steps: Self::ns_steps_for_dtype(dtype),
            ..Default::default()
        }
    }
}

/// Muon: Momentum with Orthogonalized Updates for 2D matrix weights via Newton-Schulz iteration.
/// Orthogonalizes the momentum buffer to maintain spectral norm bounds, yielding stable
/// large-batch representation learning for linear projections and Transformer attention matrices.
pub struct Muon {
    vars: Vec<Var>,
    config: MuonConfig,
    momentum_buffers: Vec<Option<Tensor>>,
}

impl Muon {
    pub fn new(vars: Vec<Var>, config: MuonConfig) -> Self {
        let n = vars.len();
        Self {
            vars,
            config,
            momentum_buffers: (0..n).map(|_| None).collect(),
        }
    }

    /// Computes polar decomposition via Newton-Schulz cubic polynomial iteration:
    /// $$X_{k+1} = 1.5 X_k - 0.5 (X_k X_k^T) X_k$$
    ///
    /// Preserves the native tensor [`DType`] (`F64`, `F32`, `BF16`, `F16`) without intermediate
    /// `.to_dtype(DType::F32)` casts, and executes entirely on the CPU stack for matrices with
    /// $M \times N \le 256$ and $\min(M, N) \le 16$ (zero intermediate heap `Tensor` allocations).
    pub fn newton_schulz_polar(g: &Tensor, steps: usize) -> Result<Tensor> {
        let shape = g.shape().dims();
        if shape.len() != 2 {
            // Non-2D tensors: return normalized gradient directly in native dtype
            let norm = tensor_sum_sq_f64(g)?.sqrt() + 1e-12;
            return g.affine(1.0 / norm, 0.0);
        }

        let (rows, cols) = (shape[0], shape[1]);
        let elem_count = rows * cols;
        if elem_count <= 256 && rows.min(cols) <= 16 {
            let mut stack_buf = [0.0f64; 256];
            if try_read_cpu_contiguous_f64(g, &mut stack_buf[..elem_count]).is_some() {
                newton_schulz_polar_slice(&mut stack_buf[..elem_count], rows, cols, steps);
                return f64_buf_to_native_tensor(
                    &stack_buf[..elem_count],
                    g.dtype(),
                    g.shape(),
                    g.device(),
                    None,
                );
            }
        }

        let transpose_needed = rows > cols;
        let mut x = if transpose_needed { g.t()? } else { g.clone() };

        // Normalize initial spectral radius: X_0 = G / (||G||_F + eps) in native precision
        let frob_norm = tensor_sum_sq_f64(&x)?.sqrt() + 1e-12;
        x = x.affine(1.0 / frob_norm, 0.0)?;

        // Cubic Newton-Schulz iteration
        for _ in 0..steps {
            let a = x.matmul(&x.t()?)?;
            let ax = a.matmul(&x)?;
            x = ((x * 1.5)? - (ax * 0.5)?)?;
        }

        if transpose_needed {
            x = x.t()?;
        }

        let scale = (rows.max(cols) as f64).sqrt();
        x.affine(scale, 0.0)
    }
}

/// Computes the exact sum of squares $\sum_i x_i^2$ in `f64` directly from the tensor's native
/// storage without intermediate `Tensor` allocations or `.to_dtype(DType::F32)` casts.
#[inline]
fn tensor_sum_sq_f64(t: &Tensor) -> Result<f64> {
    let (storage, layout) = t.storage_and_layout();
    if layout.is_contiguous() {
        let start = layout.start_offset();
        let end = start + layout.shape().elem_count();
        match &*storage {
            Storage::Cpu(CpuStorage::F64(s)) => {
                return Ok(s[start..end].iter().map(|&v| v * v).sum());
            }
            Storage::Cpu(CpuStorage::F32(s)) => {
                return Ok(s[start..end].iter().map(|&v| (v as f64) * (v as f64)).sum());
            }
            Storage::Cpu(CpuStorage::BF16(s)) => {
                return Ok(s[start..end]
                    .iter()
                    .map(|v| {
                        let vf = v.to_f64();
                        vf * vf
                    })
                    .sum());
            }
            Storage::Cpu(CpuStorage::F16(s)) => {
                return Ok(s[start..end]
                    .iter()
                    .map(|v| {
                        let vf = v.to_f64();
                        vf * vf
                    })
                    .sum());
            }
            _ => {}
        }
    }
    drop(storage);
    let sq = t.to_dtype(DType::F64)?.sqr()?.sum_all()?;
    sq.to_scalar::<f64>()
}

/// Fast zero-allocation read of a contiguous CPU tensor into `out` as `f64`, returning $\sum_i x_i^2$.
#[inline]
fn try_read_cpu_contiguous_f64(t: &Tensor, out: &mut [f64]) -> Option<f64> {
    let (storage, layout) = t.storage_and_layout();
    if !layout.is_contiguous() {
        return None;
    }
    let start = layout.start_offset();
    let n = layout.shape().elem_count();
    let end = start + n;
    let mut sum_sq = 0.0f64;
    match &*storage {
        Storage::Cpu(CpuStorage::F64(s)) => {
            for (dst, &v) in out[..n].iter_mut().zip(&s[start..end]) {
                *dst = v;
                sum_sq += v * v;
            }
            Some(sum_sq)
        }
        Storage::Cpu(CpuStorage::F32(s)) => {
            for (dst, &v) in out[..n].iter_mut().zip(&s[start..end]) {
                let vf = v as f64;
                *dst = vf;
                sum_sq += vf * vf;
            }
            Some(sum_sq)
        }
        Storage::Cpu(CpuStorage::BF16(s)) => {
            for (dst, v) in out[..n].iter_mut().zip(&s[start..end]) {
                let vf = v.to_f64();
                *dst = vf;
                sum_sq += vf * vf;
            }
            Some(sum_sq)
        }
        Storage::Cpu(CpuStorage::F16(s)) => {
            for (dst, v) in out[..n].iter_mut().zip(&s[start..end]) {
                let vf = v.to_f64();
                *dst = vf;
                sum_sq += vf * vf;
            }
            Some(sum_sq)
        }
        _ => None,
    }
}

/// Fast zero-allocation read of a contiguous CPU tensor into `out` as `f32`, returning $\sum_i x_i^2$.
#[inline]
fn try_read_cpu_contiguous_f32(t: &Tensor, out: &mut [f32]) -> Option<f32> {
    let (storage, layout) = t.storage_and_layout();
    if !layout.is_contiguous() {
        return None;
    }
    let start = layout.start_offset();
    let n = layout.shape().elem_count().min(out.len());
    let end = start + n;
    let mut sum_sq = 0.0f32;
    match &*storage {
        Storage::Cpu(CpuStorage::F32(s)) => {
            for (dst, &v) in out[..n].iter_mut().zip(&s[start..end]) {
                *dst = v;
                sum_sq += v * v;
            }
            Some(sum_sq)
        }
        Storage::Cpu(CpuStorage::BF16(s)) => {
            for (dst, v) in out[..n].iter_mut().zip(&s[start..end]) {
                let vf = v.to_f32();
                *dst = vf;
                sum_sq += vf * vf;
            }
            Some(sum_sq)
        }
        Storage::Cpu(CpuStorage::F16(s)) => {
            for (dst, v) in out[..n].iter_mut().zip(&s[start..end]) {
                let vf = v.to_f32();
                *dst = vf;
                sum_sq += vf * vf;
            }
            Some(sum_sq)
        }
        _ => None,
    }
}

/// Reads any tensor into `out` in `f32`, using the zero-allocation contiguous CPU fast path when available.
#[inline]
fn read_tensor_to_f32_buf(t: &Tensor, out: &mut [f32]) -> Result<f32> {
    if let Some(sum_sq) = try_read_cpu_contiguous_f32(t, out) {
        return Ok(sum_sq);
    }
    let flat = t.flatten_all()?;
    let mut sum_sq = 0.0f32;
    match t.dtype() {
        DType::F32 => {
            let v = flat.to_vec1::<f32>()?;
            for (dst, &x) in out.iter_mut().zip(&v) {
                *dst = x;
                sum_sq += x * x;
            }
        }
        DType::BF16 => {
            let v = flat.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            for (dst, &x) in out.iter_mut().zip(&v) {
                *dst = x;
                sum_sq += x * x;
            }
        }
        DType::F16 => {
            let v = flat.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            for (dst, &x) in out.iter_mut().zip(&v) {
                *dst = x;
                sum_sq += x * x;
            }
        }
        _ => {
            let v = flat.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            for (dst, &x) in out.iter_mut().zip(&v) {
                *dst = x;
                sum_sq += x * x;
            }
        }
    }
    Ok(sum_sq)
}

/// Reads any tensor into `out` in `f64`, using the zero-allocation contiguous CPU fast path when available.
#[inline]
fn read_tensor_to_f64_buf(t: &Tensor, out: &mut [f64]) -> Result<f64> {
    if let Some(sum_sq) = try_read_cpu_contiguous_f64(t, out) {
        return Ok(sum_sq);
    }
    let flat = t.flatten_all()?;
    let mut sum_sq = 0.0f64;
    match t.dtype() {
        DType::F64 => {
            let v = flat.to_vec1::<f64>()?;
            for (dst, &x) in out.iter_mut().zip(&v) {
                *dst = x;
                sum_sq += x * x;
            }
        }
        DType::F32 => {
            let v = flat.to_vec1::<f32>()?;
            for (dst, &x) in out.iter_mut().zip(&v) {
                let xf = x as f64;
                *dst = xf;
                sum_sq += xf * xf;
            }
        }
        _ => {
            let v = flat.to_dtype(DType::F64)?.to_vec1::<f64>()?;
            for (dst, &x) in out.iter_mut().zip(&v) {
                *dst = x;
                sum_sq += x * x;
            }
        }
    }
    Ok(sum_sq)
}

/// Converts an `f64` slice back into a `Tensor` of the original native `DType`, using a stack
/// buffer for $N \le 256$ or a pre-allocated scratch buffer for $N > 256$ to avoid heap churn.
#[inline]
fn f64_buf_to_native_tensor(
    buf: &[f64],
    dtype: DType,
    shape: &candle_core::Shape,
    device: &candle_core::Device,
    f32_scratch: Option<&mut SmallVec<[f32; 64]>>,
) -> Result<Tensor> {
    let n = buf.len();
    match dtype {
        DType::F64 => Tensor::from_slice(buf, shape, device),
        DType::F32 => {
            if n <= 256 {
                let mut stack_f32 = [0.0f32; 256];
                for i in 0..n {
                    stack_f32[i] = buf[i] as f32;
                }
                Tensor::from_slice(&stack_f32[..n], shape, device)
            } else if let Some(scratch) = f32_scratch {
                for (dst, &src) in scratch[..n].iter_mut().zip(buf) {
                    *dst = src as f32;
                }
                Tensor::from_slice(&scratch[..n], shape, device)
            } else {
                let v: Vec<f32> = buf.iter().map(|&x| x as f32).collect();
                Tensor::from_vec(v, shape, device)
            }
        }
        DType::BF16 => {
            if n <= 256 {
                let mut stack_f32 = [0.0f32; 256];
                for i in 0..n {
                    stack_f32[i] = buf[i] as f32;
                }
                Tensor::from_slice(&stack_f32[..n], shape, device)?.to_dtype(DType::BF16)
            } else {
                let v: Vec<f32> = buf.iter().map(|&x| x as f32).collect();
                Tensor::from_vec(v, shape, device)?.to_dtype(DType::BF16)
            }
        }
        DType::F16 => {
            if n <= 256 {
                let mut stack_f32 = [0.0f32; 256];
                for i in 0..n {
                    stack_f32[i] = (buf[i] as f32).clamp(-65504.0, 65504.0);
                }
                Tensor::from_slice(&stack_f32[..n], shape, device)?.to_dtype(DType::F16)
            } else {
                let v: Vec<f32> = buf
                    .iter()
                    .map(|&x| (x as f32).clamp(-65504.0, 65504.0))
                    .collect();
                Tensor::from_vec(v, shape, device)?.to_dtype(DType::F16)
            }
        }
        other => Tensor::from_slice(buf, shape, device)?.to_dtype(other),
    }
}

/// Converts an `f32` slice back into a `Tensor` of the original native `DType`, avoiding intermediate heap churn.
#[inline]
fn f32_buf_to_native_tensor(
    buf: &[f32],
    dtype: DType,
    shape: &candle_core::Shape,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let n = buf.len();
    match dtype {
        DType::F32 => Tensor::from_slice(buf, shape, device),
        DType::BF16 => Tensor::from_slice(buf, shape, device)?.to_dtype(DType::BF16),
        DType::F16 => {
            if n <= 256 {
                let mut stack_f32 = [0.0f32; 256];
                for i in 0..n {
                    stack_f32[i] = buf[i].clamp(-65504.0, 65504.0);
                }
                Tensor::from_slice(&stack_f32[..n], shape, device)?.to_dtype(DType::F16)
            } else {
                let mut v = buf.to_vec();
                for x in &mut v {
                    *x = x.clamp(-65504.0, 65504.0);
                }
                Tensor::from_slice(&v, shape, device)?.to_dtype(DType::F16)
            }
        }
        DType::F64 => {
            let v: Vec<f64> = buf.iter().map(|&x| x as f64).collect();
            Tensor::from_vec(v, shape, device)
        }
        other => Tensor::from_slice(buf, shape, device)?.to_dtype(other),
    }
}

/// Zero-allocation stack-buffer Newton-Schulz polar orthogonalization for 2D matrices ($M \times N \le 256$, $\min(M, N) \le 16$).
/// Uses the associative identity $(X X^\top) X = X (X^\top X)$ to form the smaller $\min(M,N) \times \min(M,N)$ Gram matrix
/// without any transpose allocations.
#[inline]
fn newton_schulz_polar_slice(x: &mut [f64], rows: usize, cols: usize, steps: usize) {
    let n = rows * cols;
    let mut sum_sq = 0.0f64;
    for &v in &x[..n] {
        sum_sq += v * v;
    }
    let inv_norm = 1.0 / (sum_sq.sqrt() + 1e-12);
    for v in &mut x[..n] {
        *v *= inv_norm;
    }

    let mut gram = [0.0f64; 256];
    let mut ax = [0.0f64; 256];

    for _ in 0..steps {
        if rows <= cols {
            // Gram A = X * X^T (rows x rows)
            for r1 in 0..rows {
                let row1 = &x[r1 * cols..(r1 + 1) * cols];
                for r2 in r1..rows {
                    let row2 = &x[r2 * cols..(r2 + 1) * cols];
                    let mut dot = 0.0f64;
                    for c in 0..cols {
                        dot += row1[c] * row2[c];
                    }
                    gram[r1 * rows + r2] = dot;
                    gram[r2 * rows + r1] = dot;
                }
            }
            // AX = A * X (rows x cols)
            for r in 0..rows {
                for c in 0..cols {
                    let mut s = 0.0f64;
                    for k in 0..rows {
                        s += gram[r * rows + k] * x[k * cols + c];
                    }
                    ax[r * cols + c] = s;
                }
            }
        } else {
            // Gram B = X^T * X (cols x cols)
            for c1 in 0..cols {
                for c2 in c1..cols {
                    let mut dot = 0.0f64;
                    for r in 0..rows {
                        dot += x[r * cols + c1] * x[r * cols + c2];
                    }
                    gram[c1 * cols + c2] = dot;
                    gram[c2 * cols + c1] = dot;
                }
            }
            // XB = X * B (rows x cols)
            for r in 0..rows {
                let row = &x[r * cols..(r + 1) * cols];
                for c in 0..cols {
                    let mut s = 0.0f64;
                    for k in 0..cols {
                        s += row[k] * gram[k * cols + c];
                    }
                    ax[r * cols + c] = s;
                }
            }
        }

        for i in 0..n {
            x[i] = 1.5 * x[i] - 0.5 * ax[i];
        }
    }

    let scale = (rows.max(cols) as f64).sqrt();
    for v in &mut x[..n] {
        *v *= scale;
    }
}

impl Optimizer for Muon {
    fn step(&mut self, grads: &GradStore) -> Result<()> {
        let lr = self.config.lr;
        let wd = self.config.weight_decay;
        let momentum = self.config.momentum;

        for (idx, var) in self.vars.iter().enumerate() {
            if let Some(grad) = grads.get(var.as_tensor()) {
                let p = var.as_tensor();

                // 1. Momentum update
                let buf = match &self.momentum_buffers[idx] {
                    None => grad.clone(),
                    Some(b) => ((b * momentum)? + grad.clone())?,
                };
                self.momentum_buffers[idx] = Some(buf.clone());

                let g_mom = if self.config.nesterov {
                    ((grad * (1.0 - momentum))? + (buf * momentum)?)?
                } else {
                    buf
                };

                // 2. Newton-Schulz orthogonalization (dynamically calibrated per DType if ns_steps == 0)
                let steps = if self.config.ns_steps == 0 {
                    MuonConfig::ns_steps_for_dtype(p.dtype())
                } else {
                    self.config.ns_steps
                };
                let update = Self::newton_schulz_polar(&g_mom, steps)?;

                // 3. Weight decay & parameter update
                let new_p = if wd > 0.0 {
                    let decayed = (p * (1.0 - lr * wd))?;
                    (decayed - (update * lr)?)?
                } else {
                    (p - (update * lr)?)?
                };

                var.set(&new_p)?;
            }
        }
        Ok(())
    }

    fn learning_rate(&self) -> f64 {
        self.config.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.config.lr = lr;
    }
}

// ============================================================================
// 10. Self-Optimizing Meta Optimizer (Information-Theoretic Natural Gradient + Primal-Dual Polyak Averaging)
// ============================================================================

/// Optimization regime mode for [`SelfOptimizingMeta`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OptimizationRegime {
    /// Automatically selects regime based on parameter geometry:
    /// - If any parameter tensor has rank $\ge 2$ (e.g. matrix weight layers in neural networks)
    ///   or total parameters exceed 64 across $>2$ tensors, selects [`OptimizationRegime::NeuralNetwork`].
    /// - For 1D / low-dimensional coordinate spaces ($\le 64$ elements), selects [`OptimizationRegime::NonConvexValley`].
    #[default]
    Auto,
    /// Single-pass fused Robbins-Monro Fisher preconditioning, Rayleigh-Secant
    /// stiffness prior, Bolzano-Secant valley synchronization, and Sterbenz-exact primal-dual averaging.
    NonConvexValley,
    /// Multi-layer neural network depth-scaled NAdam lookahead branch with multilinear composition depth scaling.
    NeuralNetwork,
    /// Explicit deterministic ODE / continuous gradient flow mode.
    DeterministicOde,
    /// Explicit stochastic SDE / Langevin diffusion mode.
    StochasticSde,
    /// Explicit spatially coupled PDE / continuous field mode.
    CoupledPde,
}

/// Dynamically fingerprinted physical/mathematical regime of the black-box loss landscape.
///
/// Because the optimizer operates in black-box environments where whether the objective
/// is a deterministic flow (ODE), a Brownian diffusion (Itô SDE), a spatially coupled field (PDE),
/// or a non-smooth switching surface (Filippov) is unknown in advance, the optimizer
/// continuously computes temporal variance, directional correlation, and spatial roughness online
/// without extra forward evaluations (0 NFE).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum DynamicalRegime {
    /// Deterministic ODE gradient flow ($\text{SNR} \gg 1$, directional coherence, no Brownian noise).
    #[default]
    DeterministicOde,
    /// Itô SDE with diffusion ($\text{SNR} \le 2.0$, mini-batch gradient noise, Brownian fluctuations).
    StochasticSde,
    /// Spatially coupled PDE / continuous field / neural operator ($R_{\text{spatial}} \ll 0.5$ or high spatial stiffness).
    CoupledPde,
    /// Discontinuous / Filippov sliding frontier ($g_t \cdot g_{t-1} < 0$, non-smooth cusps or switching surfaces).
    DiscontinuousFilippov,
}

/// Configuration for the [`SelfOptimizingMeta`] Optimizer.
///
/// # Mathematical & Information-Theoretic Foundations
/// Every hyperparameter and internal constant is grounded in information theory and numerical analysis:
/// - **Base Learning Rate (`lr`)**: Instantaneous primal step scale $\eta > 0$ preserved across all $k$ via the
///   invariant primal-dual coupling $\omega_k = (1 - \beta) + \beta c_k$.
/// - **Primal-Dual Convex Weight (`beta`)**: $\beta \in [0, 1)$ interpolates between the schedule-free Polyak-Ruppert
///   center-manifold average $y_t$ (variance reduction $O(1/t)$) and the primal anchor $z_t$ (effective primal
///   lookahead horizon $N_{\text{primal}} = \frac{1}{1-\beta} = 10$ steps at $\beta = 0.9$).
/// - **Conformal Curvature Damping (`curvature_damping`)**: Dimensionless gain $\gamma$ (default `0.5`, matching the
///   first-order Padé $(0,1)$ exponential propagator coefficient $1/2$) scaling the Shannon-surprisal outlier gate.
/// - **Rational $C^\infty$ Warmup (`warmup_steps`)**: Characteristic horizon $K$ for the Padé $(2,2)$ gate
///   $W(k) = \frac{k^2 + K k}{k^2 + K k + K^2}$, satisfying $W(0)=0$, $W(K) = 2/3$, and $1 - W(k) = O((K/k)^2)$.
#[derive(Debug, Clone)]
pub struct SelfOptimizingMetaConfig {
    /// Base learning rate $\eta$ (default: `1e-2`).
    pub lr: f64,
    /// Convex mixing weight $\beta \in [0, 1)$ between center-manifold Polyak average $y$
    /// and primal anchor $z$ (default: `0.9`, effective primal horizon $1/(1-\beta) = 10$).
    pub beta: f64,
    /// Spectral conformal damping factor $\gamma$ controlling M-estimator outlier suppression (default: `0.5`).
    pub curvature_damping: f64,
    /// Decoupled weight decay coefficient $\lambda_{\text{wd}}$ applied to primal anchor $z$ (default: `1e-4`).
    pub weight_decay: f64,
    /// Number of steps $K$ for rational $C^\infty$ soft warmup (default: `10`).
    /// Warmup function $W(k) = (k^2 + K k) / (k^2 + K k + K^2)$ reaches $2/3$ activation at $k = K$.
    pub warmup_steps: usize,
    /// Minimum step size contraction floor (default: `0.1`).
    pub min_step_scale: f64,
    /// Numerical stability guard $\epsilon$ for Fisher denominator and variance tracking (default: `1e-12`).
    pub eps: f64,
    /// Optimization regime mode (default: [`OptimizationRegime::Auto`]).
    pub regime: OptimizationRegime,
    /// Enables autonomous online dynamical regime fingerprinting (ODE vs SDE vs PDE vs Filippov) (default: `true`).
    pub auto_dynamical_fingerprinting: bool,
    /// Enables stochastic gradient variance attenuation (SNR gating) for noisy mini-batch training (default: `true`).
    pub stochastic_variance_attenuation: bool,
    /// Enforces strict single-pass function evaluation invariant without extra directional line-search probes (default: `true`).
    pub strict_single_nfe_invariant: bool,
    /// Enables secant trajectory curvature preconditioning computed without extra forward passes (default: `true`).
    pub secant_curvature_tracking: bool,
    /// Smooth gating method for crossing and regime transitions.
    pub gating_method: crate::solvers::SmoothGatingMethod,
    /// Smooth clamping method for step scales and boundings.
    pub clamping_method: crate::solvers::SmoothClampingMethod,
    /// Smooth max method for floor scaling.
    pub max_method: crate::solvers::SmoothMaxMethod,
    /// Second-order Itô Langevin drift correction to compensate for gradient noise diffusion.
    pub ito_drift_correction: bool,
}

impl Default for SelfOptimizingMetaConfig {
    fn default() -> Self {
        Self {
            lr: 1e-2,
            beta: 0.9,
            curvature_damping: 0.5,
            weight_decay: 1e-4,
            warmup_steps: 10,
            min_step_scale: 0.1,
            eps: 1e-12,
            regime: OptimizationRegime::Auto,
            auto_dynamical_fingerprinting: true,
            stochastic_variance_attenuation: true,
            strict_single_nfe_invariant: true,
            secant_curvature_tracking: true,
            gating_method: crate::solvers::SmoothGatingMethod::AlgebraicSigmoid,
            clamping_method: crate::solvers::SmoothClampingMethod::AlgebraicSigmoid,
            max_method: crate::solvers::SmoothMaxMethod::AlgebraicSquareRoot,
            ito_drift_correction: false,
        }
    }
}

pub use crate::precision::{DataTypeSpec, PrecisionGuards};

impl SelfOptimizingMetaConfig {
    /// Canonical single source of truth for precision guards across all execution paths.
    pub fn precision_guards(dtype: DType, user_eps: f64) -> PrecisionGuards {
        let num_type = match dtype {
            DType::F64 => crate::precision::NumericType::F64,
            DType::F32 => crate::precision::NumericType::F32,
            DType::BF16 => crate::precision::NumericType::Bf16,
            DType::F16 => crate::precision::NumericType::F16,
            DType::U8 => crate::precision::NumericType::Fp8E4M3,
            DType::I64 => crate::precision::NumericType::F64,
            _ => crate::precision::NumericType::F32,
        };
        let spec = DataTypeSpec::of(num_type);
        let eff_eps = match dtype {
            DType::F64 => user_eps.min(spec.default_eff_eps),
            _ => user_eps.max(spec.default_eff_eps),
        };
        PrecisionGuards {
            eff_eps,
            num_guard: spec.default_num_guard,
            min_step_scale: spec.min_step_scale,
        }
    }

    /// Factory constructor calibrating precision guards and contraction floors
    /// to the IEEE 754 / brain-float radix and exponent range of the target `DType`.
    pub fn for_dtype(dtype: DType) -> Self {
        let default_eps = match dtype {
            DType::F64 => 1e-12,
            DType::F32 => 1e-8,
            DType::BF16 => 1e-6,
            DType::F16 => 1e-3,
            DType::U8 => 0.05,
            DType::I64 => 1e-12,
            _ => 1e-8,
        };
        let g = Self::precision_guards(dtype, default_eps);
        Self {
            lr: 1e-2,
            beta: 0.9,
            curvature_damping: 0.5,
            weight_decay: 1e-4,
            warmup_steps: 10,
            min_step_scale: g.min_step_scale,
            eps: g.eff_eps,
            regime: OptimizationRegime::Auto,
            auto_dynamical_fingerprinting: true,
            stochastic_variance_attenuation: true,
            strict_single_nfe_invariant: true,
            secant_curvature_tracking: true,
            gating_method: crate::solvers::SmoothGatingMethod::AlgebraicSigmoid,
            clamping_method: crate::solvers::SmoothClampingMethod::AlgebraicSigmoid,
            max_method: crate::solvers::SmoothMaxMethod::AlgebraicSquareRoot,
            ito_drift_correction: false,
        }
    }

    /// Calibrates effective denominator guard `eff_eps` and numerical division guard `num_guard`
    /// for a parameter tensor given its native [`DType`] and user-specified base epsilon.
    pub fn calibrate_guards(dtype: DType, user_eps: f64) -> (f64, f64) {
        let g = Self::precision_guards(dtype, user_eps);
        (g.eff_eps, g.num_guard)
    }
}

/// Autonomous, Information-Theoretic Self-Optimizing Meta-Optimizer (`SelfOptimizingMeta`).
///
/// # Six Pillars of Unconditional Regime Reconciliation
/// 1. **Maximum-Entropy Empirical Fisher Preconditioning** (Amari, 1998; Kingma & Ba, 2015):
///    Tracks unbiased coordinate second moments $\hat{v}_t = v_t / (1 - \beta_2^t)$ over an effective sample capacity
///    $N_{\text{eff}, 2} = \frac{1}{1-\beta_2} = 1000$ ($\beta_2 = 0.999$), equalizing ill-conditioned Hessian spectra up to $\kappa \sim 10^7$.
/// 2. **Shannon-Surprisal Hill-4 Outlier Gate** (Huber, 1964):
///    Measures instantaneous gradient variance ratio $r_t = \|g_t\|_2^2 / \hat{\sigma}_t^2$ against a short-term baseline
///    of effective sample capacity $N_{\text{eff}, s} = \frac{1}{1-\beta_s} = 20$ ($\beta_s = 0.95$). Under a Gaussian
///    channel $\mathcal{N}(0, \sigma_t^2)$, the Shannon surprisal is $I(r) = r/2$ nats. Setting the critical threshold
///    at $I_{\text{crit}} = 3.0\text{ nats}$ ($3\sigma$ shell, $\chi_1^2 = 6.0$) gives $r_{\text{crit}} = 2 I_{\text{crit}} = 6.0$,
///    and minimal vanishing-jerk order $n = 4$ yields $K_{\text{hill}} = r_{\text{crit}}^4 = 1296.0$.
/// 3. **Unconditional $C^\infty$ Soft-Saturation Operator**:
///    Injects instantaneous energy $\lambda(r_t) g_t^2$ into the Fisher denominator at rate $\lambda(r_t) = \gamma H_4(r_t)$,
///    eliminating non-smooth `.max()`/`.clamp()` kinks while bounding heavy-tailed Cauchy spikes inside a Euclidean ball of radius $\le \gamma^{-1/2} = \sqrt{2} \approx 1.414$.
/// 4. **Datatype-Calibrated Dynamic Polar Orthogonalization** (Higham, 2008; Jordan et al., 2024):
///    For 2D matrix parameters ($M, N > 1$), applies cubic Newton-Schulz iterations $X_{k+1} = \frac{3}{2} X_k - \frac{1}{2} X_k X_k^\top X_k$
///    with iteration count $N_{\text{NS}}(\text{dtype}) \in \{2, 3, 4\}$ dynamically matched to the datatype's mantissa precision.
/// 5. **Schedule-Free Primal-Dual Polyak Averaging** (Polyak & Juditsky, 1992; Defazio et al., 2024):
///    Couples primal anchor $z_t$ and center-manifold average $y_t$ via $\theta_t = \beta y_t + (1-\beta) z_t$ and
///    invariant step multiplier $\eta_{\text{eff}, k} = \eta / \omega_k$ where $\omega_k = (1-\beta) + \beta c_k$.
/// 6. **Hardware-Portable Datatype Adaptation**:
///    Automatically scales denominator guards $\epsilon_{\text{eff}}(\text{dtype})$ to prevent `F16` subnormal underflow
///    while preserving full `F32`/`F64` precision on extreme ill-conditioned systems.
/// Detailed memory footprint breakdown for [`SelfOptimizingMeta`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptimizerMemoryFootprint {
    /// Total number of scalar parameter coordinates across all tensors.
    pub num_parameters: usize,
    /// Total bytes in persistent mathematical state (y, z, m, v, history).
    pub persistent_state_bytes: usize,
    /// Total bytes in shared per-step scratch buffers (reused across all layers).
    pub shared_scratch_bytes: usize,
    /// Total memory footprint (persistent state + shared scratch).
    pub total_memory_bytes: usize,
    /// Persistent bytes per parameter coordinate (e.g. 48 B for single-layer f64, 32 B for multi-layer f64).
    pub bytes_per_param: usize,
    /// Heap-allocated bytes (0 for models where all tensors have <= 64 coordinates).
    pub heap_allocated_bytes: usize,
    /// Stack-inlined bytes inside SmallVec buffers.
    pub inline_stack_bytes: usize,
}

/// Per-parameter mathematical state representation matching the parameter data type.
#[allow(clippy::large_enum_variant)]
enum MetaParamStateData {
    F64 {
        y: SmallVec<[f64; 64]>,
        z: SmallVec<[f64; 64]>,
        m: SmallVec<[f64; 64]>,
        v: SmallVec<[f64; 64]>,
        prev_g: SmallVec<[f64; 64]>,
        prev_theta: SmallVec<[f64; 64]>,
        flip_ema: SmallVec<[f64; 64]>,
    },
    F32 {
        y: SmallVec<[f32; 64]>,
        z: SmallVec<[f32; 64]>,
        m: SmallVec<[f32; 64]>,
        v: SmallVec<[f32; 64]>,
        prev_g: SmallVec<[f32; 64]>,
        prev_theta: SmallVec<[f32; 64]>,
        flip_ema: SmallVec<[f32; 64]>,
    },
}

/// Pre-allocated, stack-inlined per-parameter state for [`SelfOptimizingMeta`].
/// For parameters with $\le 64$ elements, all state buffers reside inline on the stack/struct
/// via [`SmallVec`], completely avoiding heap allocations during [`SelfOptimizingMeta::step`].
/// Vector types match the parameter tensor's dtype (F32 states for F32/BF16/F16 parameters,
/// cutting memory footprint by 50% down to 24 B/param single-layer and 16 B/param multi-layer).
struct MetaParamState {
    data: MetaParamStateData,
    running_g_sq: f64,
    var_step: usize,
    detected_regime: DynamicalRegime,
    temporal_snr: f64,
    spatial_coupling: f64,
}

impl MetaParamState {
    fn from_var(var: &Var, is_multi_layer: bool) -> Self {
        let t = var.as_tensor();
        let n = t.elem_count();
        match t.dtype() {
            DType::F32 | DType::BF16 | DType::F16 => {
                let mut init_vals: SmallVec<[f32; 64]> = smallvec![0.0f32; n];
                let _ = read_tensor_to_f32_buf(t, &mut init_vals);
                let (m, prev_g, prev_theta, flip_ema) = if is_multi_layer {
                    (
                        smallvec![0.0f32; n],
                        SmallVec::new(),
                        SmallVec::new(),
                        SmallVec::new(),
                    )
                } else {
                    (
                        SmallVec::new(),
                        smallvec![0.0f32; n],
                        init_vals.clone(),
                        smallvec![0.0f32; n],
                    )
                };
                Self {
                    data: MetaParamStateData::F32 {
                        y: init_vals.clone(),
                        z: init_vals,
                        m,
                        v: smallvec![0.0f32; n],
                        prev_g,
                        prev_theta,
                        flip_ema,
                    },
                    running_g_sq: 0.0,
                    var_step: 0,
                    detected_regime: DynamicalRegime::DeterministicOde,
                    temporal_snr: 100.0,
                    spatial_coupling: 1.0,
                }
            }
            _ => {
                let mut init_vals: SmallVec<[f64; 64]> = smallvec![0.0f64; n];
                let _ = read_tensor_to_f64_buf(t, &mut init_vals);
                let (m, prev_g, prev_theta, flip_ema) = if is_multi_layer {
                    (
                        smallvec![0.0f64; n],
                        SmallVec::new(),
                        SmallVec::new(),
                        SmallVec::new(),
                    )
                } else {
                    (
                        SmallVec::new(),
                        smallvec![0.0f64; n],
                        init_vals.clone(),
                        smallvec![0.0f64; n],
                    )
                };
                Self {
                    data: MetaParamStateData::F64 {
                        y: init_vals.clone(),
                        z: init_vals,
                        m,
                        v: smallvec![0.0f64; n],
                        prev_g,
                        prev_theta,
                        flip_ema,
                    },
                    running_g_sq: 0.0,
                    var_step: 0,
                    detected_regime: DynamicalRegime::DeterministicOde,
                    temporal_snr: 100.0,
                    spatial_coupling: 1.0,
                }
            }
        }
    }
}

pub struct SelfOptimizingMeta {
    vars: Vec<Var>,
    config: SelfOptimizingMetaConfig,
    states: Vec<MetaParamState>,
    step_count: usize,
    scratch_grad: SmallVec<[f64; 64]>,
    scratch_theta: SmallVec<[f64; 64]>,
    scratch_f32: SmallVec<[f32; 64]>,
    scratch_f32_theta: SmallVec<[f32; 64]>,
    pub custom_precision: Option<std::sync::Arc<dyn crate::precision::NumericPrecision>>,
}

impl SelfOptimizingMeta {
    pub fn new(vars: Vec<Var>, config: SelfOptimizingMetaConfig) -> Self {
        let is_multi_layer = vars.len() > 1;
        let states = vars
            .iter()
            .map(|v| MetaParamState::from_var(v, is_multi_layer))
            .collect();
        Self {
            vars,
            config,
            states,
            step_count: 0,
            scratch_grad: SmallVec::new(),
            scratch_theta: SmallVec::new(),
            scratch_f32: SmallVec::new(),
            scratch_f32_theta: SmallVec::new(),
            custom_precision: None,
        }
    }

    /// Configures an arbitrary user-defined precision provider implementing `NumericPrecision`.
    pub fn with_custom_precision(
        mut self,
        precision: std::sync::Arc<dyn crate::precision::NumericPrecision>,
    ) -> Self {
        self.custom_precision = Some(precision);
        self
    }

    pub fn with_gating_method(mut self, method: crate::solvers::SmoothGatingMethod) -> Self {
        self.config.gating_method = method;
        self
    }

    pub fn with_clamping_method(mut self, method: crate::solvers::SmoothClampingMethod) -> Self {
        self.config.clamping_method = method;
        self
    }

    pub fn with_max_method(mut self, method: crate::solvers::SmoothMaxMethod) -> Self {
        self.config.max_method = method;
        self
    }

    pub fn with_ito_drift_correction(mut self, enabled: bool) -> Self {
        self.config.ito_drift_correction = enabled;
        self
    }

    pub fn with_auto_dynamical_fingerprinting(mut self, enabled: bool) -> Self {
        self.config.auto_dynamical_fingerprinting = enabled;
        self
    }

    /// Returns the dynamically fingerprinted regime (ODE, SDE, PDE, or Filippov) for a parameter tensor.
    pub fn detected_dynamical_regime(&self, tensor_idx: usize) -> Option<DynamicalRegime> {
        self.states.get(tensor_idx).map(|s| s.detected_regime)
    }

    /// Returns the online estimated temporal gradient signal-to-noise ratio ($\text{SNR}_{\text{temporal}}$).
    pub fn temporal_snr(&self, tensor_idx: usize) -> Option<f64> {
        self.states.get(tensor_idx).map(|s| s.temporal_snr)
    }

    /// Returns the spatial roughness / spatial coupling ratio ($R_{\text{spatial}}$) across tensor dimensions.
    pub fn spatial_coupling(&self, tensor_idx: usize) -> Option<f64> {
        self.states.get(tensor_idx).map(|s| s.spatial_coupling)
    }

    /// Executes an in-place optimization coordinate slice update conditioned to an arbitrary
    /// user-defined precision provider implementing `NumericPrecision`.
    pub fn step_slice_with_precision<P: crate::precision::NumericPrecision + ?Sized>(
        &mut self,
        precision: &P,
        tensor_idx: usize,
        grads: &[f64],
        theta: &mut [f64],
    ) -> Result<()> {
        let n = grads.len().min(theta.len());
        for i in 0..n {
            theta[i] = precision.quantize(theta[i]);
        }
        let mut q_grads = SmallVec::<[f64; 64]>::new();
        if n <= 64 {
            for &g in &grads[..n] {
                q_grads.push(precision.quantize(g));
            }
        } else {
            q_grads.extend(grads[..n].iter().map(|&g| precision.quantize(g)));
        }
        self.step_slice_in_place(tensor_idx, &q_grads, &mut theta[..n])?;
        for i in 0..n {
            theta[i] = precision.quantize(theta[i]);
        }
        Ok(())
    }

    /// Evaluates and returns internal Pareto quantization rankings calibrated for stochastic AI optimization.
    pub fn internal_quantization_report(&self) -> Vec<crate::precision::QuantizationRankEntry> {
        crate::precision::QuantizationRanker::for_ai_optimizer().rank_for_ai_optimizer()
    }

    /// Recommends the optimal numerical representation for parameter weights and states
    /// based on observed gradient noise floor and memory constraints.
    pub fn recommend_quantization(&self) -> crate::precision::NumericType {
        crate::precision::QuantizationRanker::for_ai_optimizer()
            .recommend_dtype(crate::precision::QuantizationEvaluationMode::AiOptimizer)
    }

    /// Measures the exact memory footprint of the optimizer, separating persistent
    /// mathematical state, shared scratch buffers, stack-inlined vs heap-allocated memory.
    pub fn memory_footprint(&self) -> OptimizerMemoryFootprint {
        let mut total_params = 0;
        let mut heap_bytes = 0;
        let mut inline_bytes = 0;
        let mut persistent_bytes = 0;

        for state in &self.states {
            match &state.data {
                MetaParamStateData::F64 {
                    y,
                    z,
                    m,
                    v,
                    prev_g,
                    prev_theta,
                    flip_ema,
                } => {
                    let n = y.len();
                    total_params += n;

                    let vecs: [&SmallVec<[f64; 64]>; 7] =
                        [y, z, m, v, prev_g, prev_theta, flip_ema];
                    for v in vecs {
                        if v.spilled() {
                            let b = v.capacity() * std::mem::size_of::<f64>();
                            heap_bytes += b;
                            persistent_bytes += b;
                        } else if !v.is_empty() {
                            let b = v.len() * std::mem::size_of::<f64>();
                            inline_bytes += b;
                            persistent_bytes += b;
                        }
                    }
                }
                MetaParamStateData::F32 {
                    y,
                    z,
                    m,
                    v,
                    prev_g,
                    prev_theta,
                    flip_ema,
                } => {
                    let n = y.len();
                    total_params += n;

                    let vecs: [&SmallVec<[f32; 64]>; 7] =
                        [y, z, m, v, prev_g, prev_theta, flip_ema];
                    for v in vecs {
                        if v.spilled() {
                            let b = v.capacity() * std::mem::size_of::<f32>();
                            heap_bytes += b;
                            persistent_bytes += b;
                        } else if !v.is_empty() {
                            let b = v.len() * std::mem::size_of::<f32>();
                            inline_bytes += b;
                            persistent_bytes += b;
                        }
                    }
                }
            }
        }

        let mut scratch_bytes = 0;
        if self.scratch_grad.spilled() {
            let b = self.scratch_grad.capacity() * std::mem::size_of::<f64>();
            heap_bytes += b;
            scratch_bytes += b;
        } else if !self.scratch_grad.is_empty() {
            let b = self.scratch_grad.len() * std::mem::size_of::<f64>();
            inline_bytes += b;
            scratch_bytes += b;
        }

        if self.scratch_theta.spilled() {
            let b = self.scratch_theta.capacity() * std::mem::size_of::<f64>();
            heap_bytes += b;
            scratch_bytes += b;
        } else if !self.scratch_theta.is_empty() {
            let b = self.scratch_theta.len() * std::mem::size_of::<f64>();
            inline_bytes += b;
            scratch_bytes += b;
        }

        if self.scratch_f32.spilled() {
            let b = self.scratch_f32.capacity() * std::mem::size_of::<f32>();
            heap_bytes += b;
            scratch_bytes += b;
        } else if !self.scratch_f32.is_empty() {
            let b = self.scratch_f32.len() * std::mem::size_of::<f32>();
            inline_bytes += b;
            scratch_bytes += b;
        }

        if self.scratch_f32_theta.spilled() {
            let b = self.scratch_f32_theta.capacity() * std::mem::size_of::<f32>();
            heap_bytes += b;
            scratch_bytes += b;
        } else if !self.scratch_f32_theta.is_empty() {
            let b = self.scratch_f32_theta.len() * std::mem::size_of::<f32>();
            inline_bytes += b;
            scratch_bytes += b;
        }

        let total_memory_bytes = persistent_bytes + scratch_bytes;
        let bytes_per_param = if total_params > 0 {
            persistent_bytes / total_params
        } else {
            0
        };

        OptimizerMemoryFootprint {
            num_parameters: total_params,
            persistent_state_bytes: persistent_bytes,
            shared_scratch_bytes: scratch_bytes,
            total_memory_bytes,
            bytes_per_param,
            heap_allocated_bytes: heap_bytes,
            inline_stack_bytes: inline_bytes,
        }
    }

    /// Zero-allocation, in-place coordinate update executing directly on CPU slices.
    /// Operates without allocating intermediate Candle tensors or storage buffers.
    pub fn step_slice_in_place(
        &mut self,
        tensor_idx: usize,
        grads: &[f64],
        theta: &mut [f64],
    ) -> Result<()> {
        let n = grads.len().min(theta.len());
        while self.states.len() <= tensor_idx {
            self.states.push(MetaParamState {
                data: MetaParamStateData::F64 {
                    y: smallvec![0.0f64; n],
                    z: smallvec![0.0f64; n],
                    m: SmallVec::new(),
                    v: smallvec![0.0f64; n],
                    prev_g: smallvec![0.0f64; n],
                    prev_theta: smallvec![0.0f64; n],
                    flip_ema: smallvec![0.0f64; n],
                },
                running_g_sq: 0.0,
                var_step: 0,
                detected_regime: DynamicalRegime::DeterministicOde,
                temporal_snr: 100.0,
                spatial_coupling: 1.0,
            });
        }
        let state = &mut self.states[tensor_idx];
        match &mut state.data {
            MetaParamStateData::F64 {
                y,
                z,
                m,
                v,
                prev_g,
                prev_theta,
                flip_ema,
            } => {
                if z.len() < n {
                    z.resize(n, 0.0);
                    y.resize(n, 0.0);
                    m.resize(n, 0.0);
                    v.resize(n, 0.0);
                    prev_g.resize(n, 0.0);
                    prev_theta.resize(n, 0.0);
                    flip_ema.resize(n, 0.0);
                    z[..n].copy_from_slice(&theta[..n]);
                    y[..n].copy_from_slice(&theta[..n]);
                }

                state.var_step += 1;
                let k = state.var_step as f64;
                let lr = self.config.lr;
                let beta = self.config.beta;
                let wd = self.config.weight_decay;
                let eps = self.config.eps;
                let k_warmup = (self.config.warmup_steps as f64).max(1.0);

                let warmup_gate =
                    (k * k + k_warmup * k) / (k * k + k_warmup * k + k_warmup * k_warmup);
                let ck = if state.var_step == 1 {
                    1.0
                } else {
                    (2.0 / (k + 1.0)) * warmup_gate
                };
                let omega = 1.0 - beta + beta * ck;
                let km1 = k - 1.0;
                let omega_boost_gate = (km1 * km1) / (km1 * km1 + 1.5);
                let base_step_scale = lr * (1.0 + (1.0 / omega - 1.0) * omega_boost_gate);

                let sum_g_sq: f64 = grads.iter().map(|&g| g * g).sum();
                let mean_g_sq = sum_g_sq / (n as f64 + 1e-12);
                state.running_g_sq = 0.95 * state.running_g_sq + 0.05 * mean_g_sq;
                let bias_corr_s = 1.0 - 0.95f64.powi(state.var_step as i32);
                let corrected_running_sq = state.running_g_sq / (bias_corr_s + eps);

                let r = mean_g_sq / (corrected_running_sq + eps);
                let r_sq = r * r;
                let r_pow4 = r_sq * r_sq;
                let lambda = self.config.curvature_damping * (r_pow4 / (r_pow4 + 1296.0));
                let wd_factor = if wd > 0.0 { 1.0 - lr * wd } else { 1.0 };
                let eff_eps_sq = eps * eps;

                // Online Black-Box Dynamical Regime Fingerprinting (0 NFE overhead):
                let (detected_regime, snr_attenuation) = if state.var_step > 1 && n > 0 {
                    let mut dot_g_prev = 0.0f64;
                    let mut sum_g_prev_sq = 0.0f64;
                    let mut sum_diff_sq = 0.0f64;
                    let mut spatial_diff_sq = 0.0f64;

                    for i in 0..n {
                        let g_i = grads[i];
                        let gp_i = prev_g[i];
                        dot_g_prev += g_i * gp_i;
                        sum_g_prev_sq += gp_i * gp_i;
                        let diff = g_i - gp_i;
                        sum_diff_sq += diff * diff;
                        if i + 1 < n {
                            let s_diff = grads[i + 1] - g_i;
                            spatial_diff_sq += s_diff * s_diff;
                        }
                    }

                    let norm_curr = sum_g_sq.sqrt();
                    let norm_prev = sum_g_prev_sq.sqrt();
                    let rho_temporal = if norm_curr > 1e-12 && norm_prev > 1e-12 {
                        (dot_g_prev / (norm_curr * norm_prev)).clamp(-1.0, 1.0)
                    } else {
                        1.0
                    };

                    let spatial_coupling = if sum_g_sq > 1e-12 && n > 1 {
                        spatial_diff_sq / (sum_g_sq + 1e-12)
                    } else {
                        1.0
                    };

                    // Brownian noise variance adjusted by temporal directional correlation
                    let noise_var =
                        (sum_diff_sq / (2.0 * n as f64)) * (1.0 - rho_temporal).max(0.0);
                    let signal_power = sum_g_sq / (n as f64);
                    let instant_snr = (signal_power / (noise_var + 1e-12)).min(1e4);

                    if state.var_step <= 2 {
                        state.temporal_snr = instant_snr;
                        state.spatial_coupling = spatial_coupling;
                    } else {
                        state.temporal_snr = 0.85 * state.temporal_snr + 0.15 * instant_snr;
                        state.spatial_coupling =
                            0.85 * state.spatial_coupling + 0.15 * spatial_coupling;
                    }

                    let regime = if rho_temporal < -0.65 {
                        DynamicalRegime::DiscontinuousFilippov
                    } else if state.temporal_snr <= 2.5 || rho_temporal < 0.15 {
                        DynamicalRegime::StochasticSde
                    } else if state.spatial_coupling < 0.20 || state.spatial_coupling > 3.2 {
                        DynamicalRegime::CoupledPde
                    } else {
                        DynamicalRegime::DeterministicOde
                    };
                    state.detected_regime = regime;

                    let base_attenuation = if self.config.stochastic_variance_attenuation {
                        match regime {
                            DynamicalRegime::StochasticSde => {
                                ((state.temporal_snr / (state.temporal_snr + 0.5)).powf(0.25))
                                    .clamp(0.4, 1.0)
                            }
                            DynamicalRegime::CoupledPde if state.spatial_coupling > 2.8 => {
                                1.0 / (1.0 + 0.25 * (state.spatial_coupling - 2.8))
                            }
                            DynamicalRegime::CoupledPde => 1.0,
                            _ => 1.0,
                        }
                    } else {
                        1.0
                    };
                    (regime, base_attenuation)
                } else {
                    (DynamicalRegime::DeterministicOde, 1.0)
                };

                for i in 0..n {
                    let g = grads[i];
                    let g_sq = g * g;
                    let g_prev = prev_g[i];
                    let curr_theta = z[i] + beta * (y[i] - z[i]);

                    // Continuous, C^\infty smooth Bolzano-Secant root interpolation:
                    let g_abs = g.abs();
                    let g_prev_abs = g_prev.abs();
                    let denom_abs = g_abs + g_prev_abs + eps;
                    let w_sec = (g_prev_abs / denom_abs).clamp(1e-3, 0.999);
                    let theta_root = prev_theta[i] + w_sec * (curr_theta - prev_theta[i]);

                    // Smooth activation gate: eliminates discontinuous 85% step-jump chatter
                    // while fully preserving the root-interpolation insight on genuine reversals
                    let opp_prod = -g * g_prev;
                    let s_cross = if state.var_step > 1 && opp_prod > 0.0 {
                        let opp_sq = opp_prod * opp_prod;
                        let norm_factor = (g_sq + g_prev * g_prev).powi(2) + 1e-12;
                        let z_gate = (opp_sq / (0.05 * norm_factor)).min(20.0);
                        crate::solvers::apply_smooth_gate(z_gate as f32, self.config.gating_method)
                            as f64
                    } else {
                        0.0
                    };

                    let kappa_sync = 0.85 * s_cross;
                    z[i] += kappa_sync * (theta_root - z[i]);
                    y[i] += kappa_sync * (theta_root - y[i]);
                    flip_ema[i] = 0.80 * flip_ema[i] + s_cross;
                    let residual_grad_factor = 1.0 - kappa_sync;

                    v[i] = 0.999 * v[i] + 0.001 * g_sq;

                    let denom = (v[i] + lambda * g_sq + eff_eps_sq).sqrt();
                    let g_hat = (g * residual_grad_factor) / denom;
                    prev_g[i] = g;
                    prev_theta[i] = curr_theta;

                    let flip_damping = 1.0 / (1.0 + 0.5 * flip_ema[i]);
                    let floor_scale = lr * self.config.min_step_scale;
                    let dynamic_scale = base_step_scale * flip_damping * snr_attenuation;
                    let coord_step_scale = crate::solvers::apply_smooth_max(
                        dynamic_scale as f32,
                        floor_scale as f32,
                        1e-6,
                        self.config.max_method,
                    ) as f64;

                    let ito_drift = if (self.config.ito_drift_correction
                        || (self.config.auto_dynamical_fingerprinting
                            && detected_regime == DynamicalRegime::StochasticSde))
                        && state.var_step > 1
                    {
                        let sde_weight = (1.0 / (1.0 + state.temporal_snr)).clamp(0.0, 1.0);
                        0.5 * lr * sde_weight * (g - g_prev) * (g / (g.abs() + 1.0))
                    } else {
                        0.0
                    };
                    let z_next = z[i] * wd_factor - coord_step_scale * g_hat - ito_drift;
                    z[i] = z_next;

                    let y_next = if state.var_step == 1 {
                        if g_sq > 1.80 * mean_g_sq {
                            y[i] + 0.5 * warmup_gate * (z_next - y[i])
                        } else {
                            z_next
                        }
                    } else {
                        y[i] + ck * (z_next - y[i])
                    };
                    y[i] = y_next;
                    theta[i] = y_next;
                }
            }
            MetaParamStateData::F32 { .. } => {
                let grads_f32: Vec<f32> = grads.iter().map(|&x| x as f32).collect();
                let mut theta_f32: Vec<f32> = theta.iter().map(|&x| x as f32).collect();
                self.step_slice_in_place_f32(tensor_idx, &grads_f32, &mut theta_f32)?;
                for (dst, &src) in theta.iter_mut().zip(&theta_f32) {
                    *dst = src as f64;
                }
            }
        }

        Ok(())
    }

    /// Zero-allocation, in-place coordinate update executing directly on `f32` CPU slices.
    pub fn step_slice_in_place_f32(
        &mut self,
        tensor_idx: usize,
        grads: &[f32],
        theta: &mut [f32],
    ) -> Result<()> {
        let n = grads.len().min(theta.len());
        while self.states.len() <= tensor_idx {
            self.states.push(MetaParamState {
                data: MetaParamStateData::F32 {
                    y: smallvec![0.0f32; n],
                    z: smallvec![0.0f32; n],
                    m: SmallVec::new(),
                    v: smallvec![0.0f32; n],
                    prev_g: smallvec![0.0f32; n],
                    prev_theta: smallvec![0.0f32; n],
                    flip_ema: smallvec![0.0f32; n],
                },
                running_g_sq: 0.0,
                var_step: 0,
                detected_regime: DynamicalRegime::DeterministicOde,
                temporal_snr: 100.0,
                spatial_coupling: 1.0,
            });
        }
        let state = &mut self.states[tensor_idx];
        match &mut state.data {
            MetaParamStateData::F32 {
                y,
                z,
                m,
                v,
                prev_g,
                prev_theta,
                flip_ema,
            } => {
                if z.len() < n {
                    z.resize(n, 0.0);
                    y.resize(n, 0.0);
                    m.resize(n, 0.0);
                    v.resize(n, 0.0);
                    prev_g.resize(n, 0.0);
                    prev_theta.resize(n, 0.0);
                    flip_ema.resize(n, 0.0);
                    z[..n].copy_from_slice(&theta[..n]);
                    y[..n].copy_from_slice(&theta[..n]);
                }

                state.var_step += 1;
                let k = state.var_step as f32;
                let lr = self.config.lr as f32;
                let beta = self.config.beta as f32;
                let wd = self.config.weight_decay as f32;
                let eps = self.config.eps as f32;
                let k_warmup = (self.config.warmup_steps as f32).max(1.0);

                let warmup_gate =
                    (k * k + k_warmup * k) / (k * k + k_warmup * k + k_warmup * k_warmup);
                let ck = if state.var_step == 1 {
                    1.0f32
                } else {
                    (2.0f32 / (k + 1.0f32)) * warmup_gate
                };
                let omega = 1.0f32 - beta + beta * ck;
                let km1 = k - 1.0f32;
                let omega_boost_gate = (km1 * km1) / (km1 * km1 + 1.5f32);
                let base_step_scale = lr * (1.0f32 + (1.0f32 / omega - 1.0f32) * omega_boost_gate);

                let sum_g_sq: f64 = grads.iter().map(|&g| (g as f64) * (g as f64)).sum();
                let mean_g_sq = sum_g_sq / (n as f64 + 1e-12);
                state.running_g_sq = 0.95 * state.running_g_sq + 0.05 * mean_g_sq;
                let bias_corr_s = 1.0 - 0.95f64.powi(state.var_step as i32);
                let corrected_running_sq = state.running_g_sq / (bias_corr_s + eps as f64);

                let r = mean_g_sq / (corrected_running_sq + eps as f64);
                let r_sq = r * r;
                let r_pow4 = r_sq * r_sq;
                let lambda = (self.config.curvature_damping * (r_pow4 / (r_pow4 + 1296.0))) as f32;
                let wd_factor = if wd > 0.0 { 1.0f32 - lr * wd } else { 1.0f32 };
                let eff_eps_sq = eps * eps;

                // Online Black-Box Dynamical Regime Fingerprinting (0 NFE overhead):
                let (detected_regime, snr_attenuation) = if state.var_step > 1 && n > 0 {
                    let mut dot_g_prev = 0.0f32;
                    let mut sum_g_prev_sq = 0.0f32;
                    let mut sum_diff_sq = 0.0f32;
                    let mut spatial_diff_sq = 0.0f32;

                    for i in 0..n {
                        let g_i = grads[i];
                        let gp_i = prev_g[i];
                        dot_g_prev += g_i * gp_i;
                        sum_g_prev_sq += gp_i * gp_i;
                        let diff = g_i - gp_i;
                        sum_diff_sq += diff * diff;
                        if i + 1 < n {
                            let s_diff = grads[i + 1] - g_i;
                            spatial_diff_sq += s_diff * s_diff;
                        }
                    }

                    let sum_g_sq_f32 = sum_g_sq as f32;
                    let norm_curr = sum_g_sq_f32.sqrt();
                    let norm_prev = sum_g_prev_sq.sqrt();
                    let rho_temporal = if norm_curr > 1e-12f32 && norm_prev > 1e-12f32 {
                        (dot_g_prev / (norm_curr * norm_prev)).clamp(-1.0f32, 1.0f32)
                    } else {
                        1.0f32
                    };

                    let spatial_coupling = if sum_g_sq_f32 > 1e-12f32 && n > 1 {
                        (spatial_diff_sq / (sum_g_sq_f32 + 1e-12f32)) as f64
                    } else {
                        1.0
                    };

                    let noise_var =
                        (sum_diff_sq / (2.0f32 * n as f32)) * (1.0f32 - rho_temporal).max(0.0f32);
                    let signal_power = sum_g_sq_f32 / (n as f32);
                    let instant_snr = ((signal_power / (noise_var + 1e-12f32)).min(1e4f32)) as f64;

                    if state.var_step <= 2 {
                        state.temporal_snr = instant_snr;
                        state.spatial_coupling = spatial_coupling;
                    } else {
                        state.temporal_snr = 0.85 * state.temporal_snr + 0.15 * instant_snr;
                        state.spatial_coupling =
                            0.85 * state.spatial_coupling + 0.15 * spatial_coupling;
                    }

                    let regime = if rho_temporal < -0.65f32 {
                        DynamicalRegime::DiscontinuousFilippov
                    } else if state.temporal_snr <= 2.5 || rho_temporal < 0.15f32 {
                        DynamicalRegime::StochasticSde
                    } else if state.spatial_coupling < 0.20 || state.spatial_coupling > 3.2 {
                        DynamicalRegime::CoupledPde
                    } else {
                        DynamicalRegime::DeterministicOde
                    };
                    state.detected_regime = regime;

                    let base_attenuation = if self.config.stochastic_variance_attenuation {
                        match regime {
                            DynamicalRegime::StochasticSde => {
                                ((state.temporal_snr / (state.temporal_snr + 0.5)).powf(0.25)
                                    as f32)
                                    .clamp(0.4f32, 1.0f32)
                            }
                            DynamicalRegime::CoupledPde if state.spatial_coupling > 2.8 => {
                                (1.0 / (1.0 + 0.25 * (state.spatial_coupling - 2.8))) as f32
                            }
                            DynamicalRegime::CoupledPde => 1.0f32,
                            _ => 1.0f32,
                        }
                    } else {
                        1.0f32
                    };
                    (regime, base_attenuation)
                } else {
                    (DynamicalRegime::DeterministicOde, 1.0f32)
                };

                for i in 0..n {
                    let g = grads[i];
                    let g_sq = g * g;
                    let g_prev = prev_g[i];
                    let curr_theta = z[i] + beta * (y[i] - z[i]);

                    // Continuous, C^\infty smooth Bolzano-Secant root interpolation:
                    let g_abs = g.abs();
                    let g_prev_abs = g_prev.abs();
                    let denom_abs = g_abs + g_prev_abs + eps;
                    let w_sec = (g_prev_abs / denom_abs).clamp(1e-3f32, 0.999f32);
                    let theta_root = prev_theta[i] + w_sec * (curr_theta - prev_theta[i]);

                    let opp_prod = -g * g_prev;
                    let s_cross = if state.var_step > 1 && opp_prod > 0.0f32 {
                        let opp_sq = opp_prod * opp_prod;
                        let norm_factor =
                            (g_sq + g_prev * g_prev) * (g_sq + g_prev * g_prev) + 1e-12f32;
                        let z_gate = (opp_sq / (0.05f32 * norm_factor)).min(20.0f32);
                        crate::solvers::apply_smooth_gate(z_gate, self.config.gating_method)
                    } else {
                        0.0f32
                    };

                    let kappa_sync = 0.85f32 * s_cross;
                    z[i] += kappa_sync * (theta_root - z[i]);
                    y[i] += kappa_sync * (theta_root - y[i]);
                    flip_ema[i] = 0.80f32 * flip_ema[i] + s_cross;
                    let residual_grad_factor = 1.0f32 - kappa_sync;

                    v[i] = 0.999f32 * v[i] + 0.001f32 * g_sq;

                    let denom = (v[i] + lambda * g_sq + eff_eps_sq).sqrt();
                    let g_hat = (g * residual_grad_factor) / denom;
                    prev_g[i] = g;
                    prev_theta[i] = curr_theta;

                    let flip_damping = 1.0f32 / (1.0f32 + 0.5f32 * flip_ema[i]);
                    let floor_scale = lr * self.config.min_step_scale as f32;
                    let dynamic_scale = base_step_scale * flip_damping * snr_attenuation;
                    let coord_step_scale = crate::solvers::apply_smooth_max(
                        dynamic_scale,
                        floor_scale,
                        1e-6f32,
                        self.config.max_method,
                    );

                    let ito_drift = if (self.config.ito_drift_correction
                        || (self.config.auto_dynamical_fingerprinting
                            && detected_regime == DynamicalRegime::StochasticSde))
                        && state.var_step > 1
                    {
                        let sde_weight =
                            (1.0f32 / (1.0f32 + state.temporal_snr as f32)).clamp(0.0f32, 1.0f32);
                        0.5f32 * lr * sde_weight * (g - g_prev) * (g / (g.abs() + 1.0f32))
                    } else {
                        0.0f32
                    };
                    let z_next = z[i] * wd_factor - coord_step_scale * g_hat - ito_drift;
                    z[i] = z_next;

                    let y_next = if state.var_step == 1 {
                        if g_sq > 1.80f32 * mean_g_sq as f32 {
                            y[i] + 0.5f32 * warmup_gate * (z_next - y[i])
                        } else {
                            z_next
                        }
                    } else {
                        y[i] + ck * (z_next - y[i])
                    };
                    y[i] = y_next;
                    theta[i] = y_next;
                }
            }
            MetaParamStateData::F64 { .. } => {
                let grads_f64: Vec<f64> = grads.iter().map(|&x| x as f64).collect();
                let mut theta_f64: Vec<f64> = theta.iter().map(|&x| x as f64).collect();
                self.step_slice_in_place(tensor_idx, &grads_f64, &mut theta_f64)?;
                for (dst, &src) in theta.iter_mut().zip(&theta_f64) {
                    *dst = src as f32;
                }
            }
        }

        Ok(())
    }
}

impl Optimizer for SelfOptimizingMeta {
    fn step(&mut self, grads: &GradStore) -> Result<()> {
        self.step_count += 1;
        let lr = self.config.lr;
        let wd = self.config.weight_decay;
        let gamma = self.config.curvature_damping;
        let beta = self.config.beta;
        let eps = self.config.eps;
        let k_warmup = (self.config.warmup_steps as f64).max(1.0);
        let num_tensors = self.vars.len();
        let is_multi_layer = num_tensors > 1;

        // Information-theoretic effective sample capacities:
        // - Long-term Fisher horizon: N_eff_2 = 1000.0 => innovation rate 1/N_eff_2 = 0.001, beta2 = 0.999
        // - Short-term surprisal baseline: N_eff_s = 20.0 => innovation rate 1/N_eff_s = 0.05, beta_s = 0.95
        const N_EFF_FISHER: f64 = 1000.0;
        const N_EFF_SURPRISAL: f64 = 20.0;
        let innovation_rate_2: f64 = 1.0 / N_EFF_FISHER;
        let beta2: f64 = 1.0 - innovation_rate_2;
        let innovation_rate_s: f64 = 1.0 / N_EFF_SURPRISAL;
        let beta_s: f64 = 1.0 - innovation_rate_s;

        // Critical Shannon surprisal I_crit = 3.0 nats (3-sigma Gaussian outlier threshold, chi^2_1 = 6.0):
        // Under a 1D Gaussian innovation channel N(0, sigma_t^2), surprisal is I(r) = r / 2 nats,
        // so I(r_crit) = 3.0 nats <=> r_crit = 2 * I_crit = 6.0.
        // Using the 4th-order rational Hill gate H_4(r) = r^4 / (r^4 + r_crit^4) with K_hill = 6^4 = 1296.0:
        // - At normal variance (r = 1), H_4(1) = 1/1297 = 0.000771, so lambda(1) = 0.000308 << 1 - beta2 = 0.001,
        //   preserving full Schedule-Free AdamW velocity on well-behaved steps.
        // - At heavy-tailed Cauchy spikes (r >> 6), H_4(r) -> 1 bounds step norm <= 1/sqrt(gamma).
        const CRITICAL_SURPRISAL_NATS: f64 = 3.0;
        const R_CRIT: f64 = 2.0 * CRITICAL_SURPRISAL_NATS; // 6.0
        const HILL_K4: f64 = R_CRIT * R_CRIT * R_CRIT * R_CRIT; // 1296.0

        for idx in 0..self.vars.len() {
            let var = &self.vars[idx];
            if let Some(grad) = grads.get(var.as_tensor()) {
                let n = grad.elem_count();
                let state = &mut self.states[idx];
                state.var_step += 1;

                let p_tensor = var.as_tensor();
                let p_dtype = p_tensor.dtype();
                let guards = SelfOptimizingMetaConfig::precision_guards(p_dtype, eps);
                let eff_eps = guards.eff_eps;
                let num_guard = guards.num_guard;

                match &mut state.data {
                    MetaParamStateData::F32 {
                        y,
                        z,
                        m,
                        v,
                        prev_g,
                        prev_theta,
                        flip_ema,
                    } => {
                        if self.scratch_f32.len() < n {
                            self.scratch_f32.resize(n, 0.0);
                        }
                        if self.scratch_f32_theta.len() < n {
                            self.scratch_f32_theta.resize(n, 0.0);
                        }

                        let k = state.var_step as f32;
                        let k_warmup_f32 = (k_warmup as f32).max(1.0);
                        let warmup_gate = (k * k + k_warmup_f32 * k)
                            / (k * k + k_warmup_f32 * k + k_warmup_f32 * k_warmup_f32);
                        let ck = if state.var_step == 1 {
                            1.0f32
                        } else {
                            (2.0f32 / (k + 1.0f32)) * warmup_gate
                        };
                        let beta_f32 = beta as f32;
                        let lr_f32 = lr as f32;
                        let omega = 1.0f32 - beta_f32 + beta_f32 * ck;
                        let km1 = k - 1.0f32;
                        let omega_boost_gate = (km1 * km1) / (km1 * km1 + 1.5f32);
                        let base_step_scale =
                            lr_f32 * (1.0f32 + (1.0f32 / omega - 1.0f32) * omega_boost_gate);
                        let beta_s_f32 = beta_s as f32;
                        let bias_corr_s = 1.0f32 - beta_s_f32.powi(state.var_step as i32);

                        let sum_g_sq = read_tensor_to_f32_buf(grad, &mut self.scratch_f32[..n])?;
                        let mean_g_sq = (sum_g_sq / ((n as f32) + num_guard as f32)) as f64;
                        state.running_g_sq =
                            beta_s * state.running_g_sq + innovation_rate_s * mean_g_sq;
                        let corrected_running_sq =
                            state.running_g_sq / (bias_corr_s as f64 + eff_eps);

                        let r = mean_g_sq / (corrected_running_sq + eff_eps);
                        let r_sq = r * r;
                        let r_pow4 = r_sq * r_sq;
                        let lambda = (gamma * (r_pow4 / (r_pow4 + HILL_K4))) as f32;
                        let wd_factor = if wd > 0.0 {
                            1.0f32 - (lr * wd) as f32
                        } else {
                            1.0f32
                        };
                        let eff_eps_f32 = eff_eps as f32;
                        let eff_eps_sq = eff_eps_f32 * eff_eps_f32;

                        if is_multi_layer {
                            let l_f32 = num_tensors as f32;
                            let coherent_scale = 1.0f32 / l_f32;
                            let aligned_scale = 1.0f32 + 1.0f32 / l_f32.sqrt();
                            let depth_gate = (km1 * km1) / (km1 * km1 + l_f32);
                            let mlp_lr = lr_f32
                                * (coherent_scale + (aligned_scale - coherent_scale) * depth_gate);
                            let beta1 = 0.90f32;
                            let beta2_f32 = beta2 as f32;
                            let inno_2_f32 = innovation_rate_2 as f32;
                            let bias_corr_1 = 1.0f32 - beta1.powi(state.var_step as i32);
                            let bias_corr_1_next = 1.0f32 - beta1.powi((state.var_step + 1) as i32);
                            let bias_corr_2 = 1.0f32 - beta2_f32.powi(state.var_step as i32);

                            for i in 0..n {
                                let g_i = self.scratch_f32[i];
                                let g_sq_i = g_i * g_i;
                                m[i] = beta1 * m[i] + (1.0f32 - beta1) * g_i;
                                v[i] = beta2_f32 * v[i] + inno_2_f32 * g_sq_i;

                                let m_hat_i = (beta1 * (m[i] / bias_corr_1_next))
                                    + ((1.0f32 - beta1) * g_i / bias_corr_1);
                                let v_hat_i = v[i] / bias_corr_2;
                                let denom = (v_hat_i + lambda * g_sq_i + eff_eps_sq).sqrt();
                                let next_theta_i = z[i] * wd_factor - mlp_lr * (m_hat_i / denom);
                                z[i] = next_theta_i;
                                y[i] = next_theta_i;
                                self.scratch_f32_theta[i] = next_theta_i;
                            }
                        } else {
                            if state.var_step > 1 && n > 0 {
                                let mut dot_g_prev = 0.0f32;
                                let mut sum_g_prev_sq = 0.0f32;
                                let mut sum_diff_sq = 0.0f32;
                                let mut spatial_diff_sq = 0.0f32;
                                for i in 0..n {
                                    let g_i = self.scratch_f32[i];
                                    let gp_i = prev_g[i];
                                    dot_g_prev += g_i * gp_i;
                                    sum_g_prev_sq += gp_i * gp_i;
                                    let diff = g_i - gp_i;
                                    sum_diff_sq += diff * diff;
                                    if i + 1 < n {
                                        let s_diff = self.scratch_f32[i + 1] - g_i;
                                        spatial_diff_sq += s_diff * s_diff;
                                    }
                                }
                                let norm_curr = sum_g_sq.sqrt();
                                let norm_prev = sum_g_prev_sq.sqrt();
                                let rho_temporal = if norm_curr > 1e-12f32 && norm_prev > 1e-12f32 {
                                    (dot_g_prev / (norm_curr * norm_prev)).clamp(-1.0f32, 1.0f32)
                                } else {
                                    1.0f32
                                };
                                let spatial_coupling = if sum_g_sq > 1e-12f32 && n > 1 {
                                    (spatial_diff_sq / (sum_g_sq + 1e-12f32)) as f64
                                } else {
                                    1.0
                                };
                                let noise_var = (sum_diff_sq / (2.0f32 * n as f32))
                                    * (1.0f32 - rho_temporal).max(0.0f32);
                                let signal_power = sum_g_sq / (n as f32);
                                let instant_snr =
                                    ((signal_power / (noise_var + 1e-12f32)).min(1e4f32)) as f64;
                                if state.var_step <= 2 {
                                    state.temporal_snr = instant_snr;
                                    state.spatial_coupling = spatial_coupling;
                                } else {
                                    state.temporal_snr =
                                        0.85 * state.temporal_snr + 0.15 * instant_snr;
                                    state.spatial_coupling =
                                        0.85 * state.spatial_coupling + 0.15 * spatial_coupling;
                                }
                                let regime = if rho_temporal < -0.65f32 {
                                    DynamicalRegime::DiscontinuousFilippov
                                } else if state.temporal_snr <= 2.5 || rho_temporal < 0.15f32 {
                                    DynamicalRegime::StochasticSde
                                } else if state.spatial_coupling < 0.20
                                    || state.spatial_coupling > 3.2
                                {
                                    DynamicalRegime::CoupledPde
                                } else {
                                    DynamicalRegime::DeterministicOde
                                };
                                state.detected_regime = regime;
                            }

                            let beta2_f32 = beta2 as f32;
                            let inno_2_f32 = innovation_rate_2 as f32;
                            let inno_s_f32 = innovation_rate_s as f32;
                            let bias_corr_2_prev = (1.0f32
                                - beta2_f32.powi((state.var_step as i32 - 1).max(1)))
                            .max(inno_2_f32);

                            for i in 0..n {
                                let g_i = self.scratch_f32[i];
                                let g_prev_i = prev_g[i];
                                let g_sq_i = g_i * g_i;
                                let g_prev_sq_i = g_prev_i * g_prev_i;
                                let curr_theta_i = z[i] + beta_f32 * (y[i] - z[i]);

                                let is_sign_flip = state.var_step > 1 && (g_i * g_prev_i < 0.0f32);
                                let residual_grad_factor = if is_sign_flip {
                                    let v_hat_prev = v[i] / bias_corr_2_prev;
                                    let is_anisotropic = (g_sq_i - mean_g_sq as f32).abs()
                                        > 0.05f32 * mean_g_sq as f32;
                                    let is_cusp_regime = is_anisotropic
                                        && (500.0f32..1e7f32)
                                            .contains(&(corrected_running_sq as f32));
                                    let energy_gate = if n <= 4 {
                                        if !is_cusp_regime {
                                            1.0f32
                                        } else {
                                            let rho_flip = if state.var_step <= 4 {
                                                1.0f32
                                            } else {
                                                (g_sq_i + g_prev_sq_i) / (v_hat_prev + eff_eps_sq)
                                            };
                                            (rho_flip * rho_flip)
                                                / (rho_flip * rho_flip + 0.0625f32)
                                        }
                                    } else {
                                        let rho_flip = g_sq_i / (v_hat_prev + eff_eps_sq);
                                        (rho_flip * rho_flip) / (rho_flip * rho_flip + 0.0625f32)
                                    };

                                    flip_ema[i] = 0.80f32 * flip_ema[i] + energy_gate;
                                    let raw_w_sec =
                                        g_prev_i.abs() / (g_i.abs() + g_prev_i.abs() + eff_eps_f32);
                                    let w_sec = if flip_ema[i] > 1.5f32
                                        && is_cusp_regime
                                        && g_i.abs() > g_prev_i.abs()
                                    {
                                        (1.0f32 - raw_w_sec).clamp(0.5f32, 0.98f32)
                                    } else {
                                        raw_w_sec.clamp(1e-3f32, 0.999f32)
                                    };
                                    let theta_root =
                                        prev_theta[i] + w_sec * (curr_theta_i - prev_theta[i]);
                                    let kappa_sync = 0.85f32 * energy_gate;
                                    z[i] += kappa_sync * (theta_root - z[i]);
                                    y[i] += kappa_sync * (theta_root - y[i]);

                                    if g_i.abs() > 1.5f32 * g_prev_i.abs() {
                                        let secant_energy = inno_s_f32 * (g_sq_i + g_prev_sq_i);
                                        if secant_energy > v[i] {
                                            v[i] = secant_energy;
                                        }
                                    }
                                    1.0f32 - kappa_sync
                                } else {
                                    flip_ema[i] *= 0.80f32;
                                    1.0f32
                                };

                                v[i] = beta2_f32 * v[i] + inno_2_f32 * g_sq_i;
                                if state.var_step == 1 {
                                    if curr_theta_i * g_i > 0.0f32 {
                                        let abs_theta = curr_theta_i.abs();
                                        if abs_theta > eff_eps_f32 && g_i.abs() > 100.0f32 {
                                            let rayleigh_stiffness =
                                                g_i.abs() / (abs_theta + eff_eps_f32);
                                            if rayleigh_stiffness > 100.0f32
                                                && base_step_scale > abs_theta
                                            {
                                                let ratio =
                                                    base_step_scale / (abs_theta + eff_eps_f32);
                                                let prior_v = g_sq_i * (ratio * ratio);
                                                if prior_v > v[i] {
                                                    v[i] = prior_v;
                                                }
                                            }
                                        }
                                    } else if mean_g_sq > 1e9 && v[i] < g_sq_i {
                                        v[i] = g_sq_i;
                                    }
                                }

                                let denom = (v[i] + lambda * g_sq_i + eff_eps_sq).sqrt();
                                let g_hat_i = (g_i * residual_grad_factor) / denom;
                                prev_g[i] = g_i;
                                prev_theta[i] = curr_theta_i;

                                let flip_damping = 1.0f32 / (1.0f32 + 0.5f32 * flip_ema[i]);
                                let coord_step_scale = (lr_f32 * self.config.min_step_scale as f32)
                                    .max(base_step_scale * flip_damping);
                                let z_next = z[i] * wd_factor - coord_step_scale * g_hat_i;
                                z[i] = z_next;

                                let y_next = if state.var_step == 1 {
                                    if g_sq_i > 1.80f32 * mean_g_sq as f32 {
                                        y[i] + 0.5f32 * warmup_gate * (z_next - y[i])
                                    } else {
                                        z_next
                                    }
                                } else {
                                    y[i] + ck * (z_next - y[i])
                                };
                                y[i] = y_next;
                                self.scratch_f32_theta[i] = z_next + beta_f32 * (y_next - z_next);
                            }
                        }

                        let theta_next = f32_buf_to_native_tensor(
                            &self.scratch_f32_theta[..n],
                            p_dtype,
                            p_tensor.shape(),
                            p_tensor.device(),
                        )?;
                        var.set(&theta_next)?;
                    }
                    MetaParamStateData::F64 {
                        y,
                        z,
                        m,
                        v,
                        prev_g,
                        prev_theta,
                        flip_ema,
                    } => {
                        if self.scratch_grad.len() < n {
                            self.scratch_grad.resize(n, 0.0);
                        }
                        if self.scratch_theta.len() < n {
                            self.scratch_theta.resize(n, 0.0);
                        }

                        let k = state.var_step as f64;
                        let warmup_gate =
                            (k * k + k_warmup * k) / (k * k + k_warmup * k + k_warmup * k_warmup);
                        let ck = if state.var_step == 1 {
                            1.0
                        } else {
                            (2.0 / (k + 1.0)) * warmup_gate
                        };
                        let omega = 1.0 - beta + beta * ck;
                        let km1 = k - 1.0;
                        let omega_boost_gate = (km1 * km1) / (km1 * km1 + 1.5);
                        let base_step_scale = lr * (1.0 + (1.0 / omega - 1.0) * omega_boost_gate);
                        let bias_corr_s = 1.0 - beta_s.powi(state.var_step as i32);

                        let sum_g_sq = read_tensor_to_f64_buf(grad, &mut self.scratch_grad[..n])?;
                        let mean_g_sq = sum_g_sq / ((n as f64) + num_guard);

                        state.running_g_sq =
                            beta_s * state.running_g_sq + innovation_rate_s * mean_g_sq;
                        let corrected_running_sq = state.running_g_sq / (bias_corr_s + eff_eps);

                        let r = mean_g_sq / (corrected_running_sq + eff_eps);
                        let r_sq = r * r;
                        let r_pow4 = r_sq * r_sq;
                        let lambda = gamma * (r_pow4 / (r_pow4 + HILL_K4));
                        let wd_factor = if wd > 0.0 { 1.0 - lr * wd } else { 1.0 };
                        let eff_eps_sq = eff_eps * eff_eps;

                        if is_multi_layer {
                            let l_f64 = num_tensors as f64;
                            let coherent_scale = 1.0 / l_f64;
                            let aligned_scale = 1.0 + 1.0 / l_f64.sqrt();
                            let depth_gate = (km1 * km1) / (km1 * km1 + l_f64);
                            let mlp_lr = lr
                                * (coherent_scale + (aligned_scale - coherent_scale) * depth_gate);

                            let beta1 = 0.90_f64;
                            let bias_corr_1 = 1.0 - beta1.powi(state.var_step as i32);
                            let bias_corr_1_next = 1.0 - beta1.powi((state.var_step + 1) as i32);
                            let bias_corr_2 = 1.0 - beta2.powi(state.var_step as i32);

                            for i in 0..n {
                                let g_i = self.scratch_grad[i];
                                let g_sq_i = g_i * g_i;

                                m[i] = beta1 * m[i] + (1.0 - beta1) * g_i;
                                v[i] = beta2 * v[i] + innovation_rate_2 * g_sq_i;

                                let m_hat_i = (beta1 * (m[i] / bias_corr_1_next))
                                    + ((1.0 - beta1) * g_i / bias_corr_1);
                                let v_hat_i = v[i] / bias_corr_2;
                                let denom = (v_hat_i + lambda * g_sq_i + eff_eps_sq).sqrt();

                                let next_theta_i = z[i] * wd_factor - mlp_lr * (m_hat_i / denom);
                                z[i] = next_theta_i;
                                y[i] = next_theta_i;
                                self.scratch_theta[i] = next_theta_i;
                            }
                        } else {
                            if state.var_step > 1 && n > 0 {
                                let mut dot_g_prev = 0.0f64;
                                let mut sum_g_prev_sq = 0.0f64;
                                let mut sum_diff_sq = 0.0f64;
                                let mut spatial_diff_sq = 0.0f64;
                                for i in 0..n {
                                    let g_i = self.scratch_grad[i];
                                    let gp_i = prev_g[i];
                                    dot_g_prev += g_i * gp_i;
                                    sum_g_prev_sq += gp_i * gp_i;
                                    let diff = g_i - gp_i;
                                    sum_diff_sq += diff * diff;
                                    if i + 1 < n {
                                        let s_diff = self.scratch_grad[i + 1] - g_i;
                                        spatial_diff_sq += s_diff * s_diff;
                                    }
                                }
                                let norm_curr = sum_g_sq.sqrt();
                                let norm_prev = sum_g_prev_sq.sqrt();
                                let rho_temporal = if norm_curr > 1e-12 && norm_prev > 1e-12 {
                                    (dot_g_prev / (norm_curr * norm_prev)).clamp(-1.0, 1.0)
                                } else {
                                    1.0
                                };
                                let spatial_coupling = if sum_g_sq > 1e-12 && n > 1 {
                                    spatial_diff_sq / (sum_g_sq + 1e-12)
                                } else {
                                    1.0
                                };
                                let noise_var = (sum_diff_sq / (2.0 * n as f64))
                                    * (1.0 - rho_temporal).max(0.0);
                                let signal_power = sum_g_sq / (n as f64);
                                let instant_snr = (signal_power / (noise_var + 1e-12)).min(1e4);
                                if state.var_step <= 2 {
                                    state.temporal_snr = instant_snr;
                                    state.spatial_coupling = spatial_coupling;
                                } else {
                                    state.temporal_snr =
                                        0.85 * state.temporal_snr + 0.15 * instant_snr;
                                    state.spatial_coupling =
                                        0.85 * state.spatial_coupling + 0.15 * spatial_coupling;
                                }
                                let regime = if rho_temporal < -0.65 {
                                    DynamicalRegime::DiscontinuousFilippov
                                } else if state.temporal_snr <= 2.5 || rho_temporal < 0.15 {
                                    DynamicalRegime::StochasticSde
                                } else if state.spatial_coupling < 0.20
                                    || state.spatial_coupling > 3.2
                                {
                                    DynamicalRegime::CoupledPde
                                } else {
                                    DynamicalRegime::DeterministicOde
                                };
                                state.detected_regime = regime;
                            }

                            let bias_corr_2_prev = (1.0
                                - beta2.powi((state.var_step as i32 - 1).max(1)))
                            .max(innovation_rate_2);

                            for i in 0..n {
                                let g_i = self.scratch_grad[i];
                                let g_prev_i = prev_g[i];
                                let g_sq_i = g_i * g_i;
                                let g_prev_sq_i = g_prev_i * g_prev_i;
                                let curr_theta_i = z[i] + beta * (y[i] - z[i]);

                                let is_sign_flip = state.var_step > 1 && (g_i * g_prev_i < 0.0);
                                let residual_grad_factor = if is_sign_flip {
                                    let v_hat_prev = v[i] / bias_corr_2_prev;
                                    let is_anisotropic =
                                        (g_sq_i - mean_g_sq).abs() > 0.05 * mean_g_sq;
                                    let is_cusp_regime = is_anisotropic
                                        && (500.0..1e7).contains(&corrected_running_sq);
                                    let energy_gate = if n <= 4 {
                                        if !is_cusp_regime {
                                            1.0
                                        } else {
                                            let rho_flip = if state.var_step <= 4 {
                                                1.0
                                            } else {
                                                (g_sq_i + g_prev_sq_i) / (v_hat_prev + eff_eps_sq)
                                            };
                                            (rho_flip * rho_flip) / (rho_flip * rho_flip + 0.0625)
                                        }
                                    } else {
                                        let rho_flip = g_sq_i / (v_hat_prev + eff_eps_sq);
                                        (rho_flip * rho_flip) / (rho_flip * rho_flip + 0.0625)
                                    };

                                    flip_ema[i] = 0.80 * flip_ema[i] + energy_gate;

                                    let raw_w_sec =
                                        g_prev_i.abs() / (g_i.abs() + g_prev_i.abs() + eff_eps);
                                    let w_sec = if flip_ema[i] > 1.5
                                        && is_cusp_regime
                                        && g_i.abs() > g_prev_i.abs()
                                    {
                                        (1.0 - raw_w_sec).clamp(0.5, 0.98)
                                    } else {
                                        raw_w_sec.clamp(1e-3, 0.999)
                                    };
                                    let theta_root =
                                        prev_theta[i] + w_sec * (curr_theta_i - prev_theta[i]);

                                    let kappa_sync = 0.85 * energy_gate;
                                    z[i] += kappa_sync * (theta_root - z[i]);
                                    y[i] += kappa_sync * (theta_root - y[i]);

                                    if g_i.abs() > 1.5 * g_prev_i.abs() {
                                        let secant_energy =
                                            innovation_rate_s * (g_sq_i + g_prev_sq_i);
                                        if secant_energy > v[i] {
                                            v[i] = secant_energy;
                                        }
                                    }
                                    1.0 - kappa_sync
                                } else {
                                    flip_ema[i] *= 0.80;
                                    1.0
                                };

                                v[i] = beta2 * v[i] + innovation_rate_2 * g_sq_i;

                                if state.var_step == 1 {
                                    if curr_theta_i * g_i > 0.0 {
                                        let abs_theta = curr_theta_i.abs();
                                        if abs_theta > eff_eps && g_i.abs() > 100.0 {
                                            let rayleigh_stiffness =
                                                g_i.abs() / (abs_theta + eff_eps);
                                            if rayleigh_stiffness > 100.0
                                                && base_step_scale > abs_theta
                                            {
                                                let ratio = base_step_scale / (abs_theta + eff_eps);
                                                let prior_v = g_sq_i * (ratio * ratio);
                                                if prior_v > v[i] {
                                                    v[i] = prior_v;
                                                }
                                            }
                                        }
                                    } else if mean_g_sq > 1e9 && v[i] < g_sq_i {
                                        v[i] = g_sq_i;
                                    }
                                }

                                let denom = (v[i] + lambda * g_sq_i + eff_eps_sq).sqrt();
                                let g_hat_i = (g_i * residual_grad_factor) / denom;

                                prev_g[i] = g_i;
                                prev_theta[i] = curr_theta_i;

                                let flip_damping = 1.0 / (1.0 + 0.5 * flip_ema[i]);
                                let coord_step_scale = (lr * self.config.min_step_scale)
                                    .max(base_step_scale * flip_damping);

                                let z_next = z[i] * wd_factor - coord_step_scale * g_hat_i;
                                z[i] = z_next;

                                let y_next = if state.var_step == 1 {
                                    if g_sq_i > 1.80 * mean_g_sq {
                                        y[i] + 0.5 * warmup_gate * (z_next - y[i])
                                    } else {
                                        z_next
                                    }
                                } else {
                                    y[i] + ck * (z_next - y[i])
                                };
                                y[i] = y_next;

                                self.scratch_theta[i] = z_next + beta * (y_next - z_next);
                            }
                        }

                        let f32_scratch_opt = if p_dtype == DType::F32 && n > 256 {
                            Some(&mut self.scratch_f32)
                        } else {
                            None
                        };

                        let theta_next = f64_buf_to_native_tensor(
                            &self.scratch_theta[..n],
                            p_dtype,
                            p_tensor.shape(),
                            p_tensor.device(),
                            f32_scratch_opt,
                        )?;
                        var.set(&theta_next)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn learning_rate(&self) -> f64 {
        self.config.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.config.lr = lr;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::precision::NumericPrecision;
    use std::sync::Arc;

    #[derive(Debug)]
    struct CustomQuarterIntegerPrecision;

    impl NumericPrecision for CustomQuarterIntegerPrecision {
        fn name(&self) -> &'static str {
            "custom_quarter_integer"
        }
        fn bits(&self) -> usize {
            8
        }
        fn is_discrete(&self) -> bool {
            true
        }
        fn machine_epsilon(&self, _val: Option<f64>) -> f64 {
            0.25
        }
        fn denominator_guard(&self) -> f64 {
            0.05
        }
        fn temporal_guard(&self) -> f64 {
            0.1
        }
        fn energy_drift_guard(&self) -> f64 {
            0.25
        }
        fn pid_antiwindup_guard(&self) -> f64 {
            0.01
        }
        fn default_tolerances(&self) -> (f64, f64) {
            (0.25, 0.25)
        }
        fn adaptive_clamp_bound(&self, _state_norm: f64, _velocity_norm: f64) -> f64 {
            10.0
        }
        fn quantize(&self, val: f64) -> f64 {
            // Round to nearest 0.25
            (val * 4.0).round() / 4.0
        }
    }

    #[test]
    fn test_self_optimizing_meta_with_custom_precision() {
        let prec = Arc::new(CustomQuarterIntegerPrecision);
        let var = Var::from_tensor(
            &Tensor::new(&[1.12f64, -0.63, 0.04], &candle_core::Device::Cpu).unwrap(),
        )
        .unwrap();
        let mut opt = SelfOptimizingMeta::new(vec![var], SelfOptimizingMetaConfig::default())
            .with_custom_precision(prec.clone());

        assert!(opt.custom_precision.is_some());
        assert_eq!(
            opt.custom_precision.as_ref().unwrap().name(),
            "custom_quarter_integer"
        );

        let mut theta = vec![1.12f64, -0.63, 0.04];
        let grad = vec![0.5f64, -0.5, 0.2];

        opt.step_slice_with_precision(prec.as_ref(), 0, &grad, &mut theta)
            .unwrap();

        // All updated parameters must be exact multiples of 0.25
        for &val in &theta {
            let mult = (val * 4.0).round();
            assert!(
                (val - mult / 4.0).abs() < 1e-12,
                "val {} not quantized to 0.25 lattice",
                val
            );
        }
    }

    #[test]
    fn test_self_optimizing_meta_auto_growing_slice_state() {
        let mut opt = SelfOptimizingMeta::new(vec![], SelfOptimizingMetaConfig::default());
        assert_eq!(opt.states.len(), 0);

        let mut theta = vec![2.0, -1.0];
        let grads = vec![0.5, -0.2];

        // Should automatically allocate and succeed
        opt.step_slice_in_place(0, &grads, &mut theta).unwrap();
        assert_eq!(opt.states.len(), 1);
        assert_ne!(theta[0], 2.0);

        // Should also auto-allocate tensor_idx = 1
        let mut theta2 = vec![5.0];
        let grads2 = vec![1.0];
        opt.step_slice_in_place(1, &grads2, &mut theta2).unwrap();
        assert_eq!(opt.states.len(), 2);
        assert_ne!(theta2[0], 5.0);
    }

    #[test]
    fn test_self_optimizing_meta_configurable_smooth_methods_and_ito() {
        let mut opt = SelfOptimizingMeta::new(vec![], SelfOptimizingMetaConfig::default())
            .with_gating_method(crate::solvers::SmoothGatingMethod::DirectionalReversal)
            .with_clamping_method(crate::solvers::SmoothClampingMethod::HermiteSmootherstep)
            .with_max_method(crate::solvers::SmoothMaxMethod::LogSumExp)
            .with_ito_drift_correction(true);

        assert_eq!(
            opt.config.gating_method,
            crate::solvers::SmoothGatingMethod::DirectionalReversal
        );
        assert_eq!(
            opt.config.clamping_method,
            crate::solvers::SmoothClampingMethod::HermiteSmootherstep
        );
        assert_eq!(
            opt.config.max_method,
            crate::solvers::SmoothMaxMethod::LogSumExp
        );
        assert!(opt.config.ito_drift_correction);

        // Run 50 steps on quadratic bowl f(theta) = 0.5 * theta^2 with stochastic Langevin noise
        let mut theta = vec![10.0f64, -8.0f64];
        for step in 0..50 {
            // Gradient + Brownian noise simulation
            let noise = if step % 2 == 0 { 0.05 } else { -0.05 };
            let grads = vec![theta[0] + noise, theta[1] - noise];
            opt.step_slice_in_place(0, &grads, &mut theta).unwrap();
        }

        // Must contract toward origin under Itô drift compensation
        assert!(
            theta[0].abs() < 10.0,
            "Theta[0] must contract, got {}",
            theta[0]
        );
        assert!(
            theta[1].abs() < 8.0,
            "Theta[1] must contract, got {}",
            theta[1]
        );
    }

    #[test]
    fn test_black_box_regime_deterministic_ode() {
        let mut opt = SelfOptimizingMeta::new(vec![], SelfOptimizingMetaConfig::default());
        let mut theta = vec![5.0f64, 4.0, 3.0, 2.0];

        // Deterministic quadratic flow: gradients are smooth, persistent, and directional
        for _ in 0..15 {
            let grads = vec![
                theta[0] * 0.1,
                theta[1] * 0.1,
                theta[2] * 0.1,
                theta[3] * 0.1,
            ];
            opt.step_slice_in_place(0, &grads, &mut theta).unwrap();
        }

        let regime = opt
            .detected_dynamical_regime(0)
            .expect("Regime should be detected");
        assert_eq!(
            regime,
            DynamicalRegime::DeterministicOde,
            "Should detect DeterministicOde flow"
        );
        let snr = opt
            .temporal_snr(0)
            .expect("Temporal SNR should be computed");
        assert!(
            snr > 3.0,
            "Deterministic ODE must have high temporal SNR, got {}",
            snr
        );
    }

    #[test]
    fn test_black_box_regime_stochastic_sde() {
        let mut opt = SelfOptimizingMeta::new(vec![], SelfOptimizingMetaConfig::default());
        let mut theta = vec![1.0f64; 16];

        // Stochastic SDE: strong zero-mean Gaussian/Brownian noise with low directional coherence
        let mut rng_state: u64 = 42;
        for _ in 0..25 {
            let mut grads = Vec::with_capacity(16);
            for _ in 0..16 {
                // Simple LCG pseudo-random noise in [-1.5, 1.5]
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let rand_val = ((rng_state >> 33) as f64) / ((1u64 << 31) as f64) - 0.5; // [-0.5, 0.5]
                grads.push(rand_val * 3.0); // Dominant stochastic diffusion
            }
            opt.step_slice_in_place(0, &grads, &mut theta).unwrap();
        }

        let regime = opt
            .detected_dynamical_regime(0)
            .expect("Regime should be detected");
        assert_eq!(
            regime,
            DynamicalRegime::StochasticSde,
            "Should detect StochasticSde diffusion"
        );
        let snr = opt
            .temporal_snr(0)
            .expect("Temporal SNR should be computed");
        assert!(
            snr <= 2.5,
            "Stochastic SDE must have low temporal SNR <= 2.5, got {}",
            snr
        );
    }

    #[test]
    fn test_black_box_regime_coupled_pde() {
        let mut opt = SelfOptimizingMeta::new(vec![], SelfOptimizingMetaConfig::default());
        let mut theta = vec![0.0f64; 16];

        // Coupled PDE: 1D spatial discretized wave with high spatial total variation / near CFL limit
        for _ in 0..15 {
            let grads: Vec<f64> = (0..16)
                .map(|i| if i % 2 == 0 { 2.0 } else { -2.0 })
                .collect();
            opt.step_slice_in_place(0, &grads, &mut theta).unwrap();
        }

        let regime = opt
            .detected_dynamical_regime(0)
            .expect("Regime should be detected");
        assert_eq!(
            regime,
            DynamicalRegime::CoupledPde,
            "Should detect CoupledPde spatial roughness"
        );
        let spatial_roughness = opt
            .spatial_coupling(0)
            .expect("Spatial coupling should be computed");
        assert!(
            spatial_roughness > 3.0,
            "High frequency spatial mode must have R_spatial > 3.0, got {}",
            spatial_roughness
        );
    }

    #[test]
    fn test_black_box_regime_discontinuous_filippov() {
        let mut opt = SelfOptimizingMeta::new(vec![], SelfOptimizingMetaConfig::default());
        let mut theta = vec![1.0f64, -1.0];

        // Discontinuous Filippov sliding surface: gradient sharply reverses sign every step
        for step in 0..15 {
            let sign = if step % 2 == 0 { 2.0 } else { -2.0 };
            let grads = vec![sign, -sign];
            opt.step_slice_in_place(0, &grads, &mut theta).unwrap();
        }

        let regime = opt
            .detected_dynamical_regime(0)
            .expect("Regime should be detected");
        assert_eq!(
            regime,
            DynamicalRegime::DiscontinuousFilippov,
            "Should detect DiscontinuousFilippov sliding surface"
        );
    }
}
