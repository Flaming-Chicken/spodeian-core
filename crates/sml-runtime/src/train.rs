//! PyTorch 2 Parity Training Utilities & Execution Mechanics for `sml-runtime`.
//!
//! Provides zero-Python training loop utilities:
//! - Gradient Clipping: L2 spectral norm (`clip_grad_norm`) and element-wise value (`clip_grad_value`)
//! - Dynamic Loss Scaling: Mixed precision `GradScaler` with Inf/NaN backoff
//! - Model Exponential Moving Average (`ModelEMA`) for stable convergence
//! - Strongly-typed Dataset & DataLoader primitives
//! - Training telemetry & history recorder

use crate::optim::Optimizer;
use candle_core::backprop::GradStore;
use candle_core::{Result, Tensor, Var};
use rand::seq::SliceRandom;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

// ============================================================================
// 1. Gradient Clipping
// ============================================================================

/// Clips the global L2 norm of gradients across all variables.
///
/// Returns the pre-clipped global norm. If norm exceeds `max_norm`, returns
/// a new `GradStore` with scaled gradients: g_new = g * (max_norm / (total_norm + 1e-6)).
pub fn clip_grad_norm(
    vars: &[Var],
    grads: &GradStore,
    max_norm: f64,
) -> Result<(f64, GradStore)> {
    let mut total_sq_sum = 0.0f64;

    for var in vars {
        if let Some(g) = grads.get(var.as_tensor()) {
            let sq = g.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
            total_sq_sum += sq;
        }
    }

    let global_norm = total_sq_sum.sqrt();
    let clip_coef = if global_norm > max_norm {
        max_norm / (global_norm + 1e-6)
    } else {
        1.0
    };

    // Construct scaled GradStore
    let mut clipped_store = GradStore::default();
    for var in vars {
        let t = var.as_tensor();
        if let Some(g) = grads.get(t) {
            let scaled_g = if clip_coef < 1.0 {
                (g * clip_coef)?
            } else {
                g.clone()
            };
            clipped_store.insert(t, scaled_g);
        }
    }

    Ok((global_norm, clipped_store))
}

/// Clips gradients element-wise within `[-clip_value, clip_value]`.
pub fn clip_grad_value(
    vars: &[Var],
    grads: &GradStore,
    clip_value: f64,
) -> Result<GradStore> {
    let mut clipped_store = GradStore::default();
    for var in vars {
        let t = var.as_tensor();
        if let Some(g) = grads.get(t) {
            let clamped = g.clamp(-clip_value, clip_value)?;
            clipped_store.insert(t, clamped);
        }
    }
    Ok(clipped_store)
}

// ============================================================================
// 2. GradScaler (Mixed Precision Dynamic Loss Scaling)
// ============================================================================

/// Dynamic loss scaler for FP16/BF16 mixed precision training.
/// Automatically prevents numerical underflow while safely skipping overflow steps.
pub struct GradScaler {
    scale: f64,
    growth_factor: f64,
    backoff_factor: f64,
    growth_interval: usize,
    successful_steps: usize,
}

impl Default for GradScaler {
    fn default() -> Self {
        Self {
            scale: 65536.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            successful_steps: 0,
        }
    }
}

impl GradScaler {
    pub fn new(initial_scale: f64) -> Self {
        Self {
            scale: initial_scale.max(1.0),
            ..Default::default()
        }
    }

    /// Scales loss tensor prior to backward pass: loss * scale
    pub fn scale(&self, loss: &Tensor) -> Result<Tensor> {
        loss * self.scale
    }

    /// Returns the current dynamic scale value.
    pub fn get_scale(&self) -> f64 {
        self.scale
    }

    /// Unscales gradients and executes an optimizer step.
    /// Returns `true` if step succeeded, `false` if Inf/NaN detected (step was skipped and scale halved).
    pub fn unscale_and_step(
        &mut self,
        vars: &[Var],
        grads: &GradStore,
        optimizer: &mut dyn Optimizer,
    ) -> Result<bool> {
        let inv_scale = 1.0 / self.scale;
        let mut has_inf_or_nan = false;
        let mut unscaled_store = GradStore::default();

        for var in vars {
            let t = var.as_tensor();
            if let Some(g) = grads.get(t) {
                // Check for non-finite values
                let sum = g.sum_all()?.to_scalar::<f32>()?;
                if !sum.is_finite() {
                    has_inf_or_nan = true;
                    break;
                }
                let unscaled_g = (g * inv_scale)?;
                unscaled_store.insert(t, unscaled_g);
            }
        }

        if has_inf_or_nan {
            // Halve scale on overflow, skip step
            self.scale = (self.scale * self.backoff_factor).max(1.0);
            self.successful_steps = 0;
            Ok(false)
        } else {
            // Apply unscaled gradients
            optimizer.step(&unscaled_store)?;
            self.successful_steps += 1;
            if self.successful_steps >= self.growth_interval {
                self.scale *= self.growth_factor;
                self.successful_steps = 0;
            }
            Ok(true)
        }
    }
}

// ============================================================================
// 3. ModelEMA (Exponential Moving Average of Weights)
// ============================================================================

/// Maintains shadow exponential moving average weights for test-time evaluation:
/// W_ema = beta * W_ema + (1 - beta) * W
pub struct ModelEMA {
    decay: f64,
    shadow_weights: Vec<Tensor>,
}

impl ModelEMA {
    pub fn new(vars: &[Var], decay: f64) -> Result<Self> {
        let shadow_weights = vars
            .iter()
            .map(|v| v.as_tensor().copy())
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            decay: decay.clamp(0.0, 1.0),
            shadow_weights,
        })
    }

    /// Updates shadow weights using current model weights.
    pub fn update(&mut self, vars: &[Var]) -> Result<()> {
        let beta = self.decay;
        for (shadow, var) in self.shadow_weights.iter_mut().zip(vars.iter()) {
            let current = var.as_tensor();
            let updated = ((shadow.affine(beta, 0.0))? + current.affine(1.0 - beta, 0.0)?)?;
            *shadow = updated;
        }
        Ok(())
    }

    /// Applies shadow weights to model variables in-place.
    pub fn copy_to_model(&self, vars: &[Var]) -> Result<()> {
        for (shadow, var) in self.shadow_weights.iter().zip(vars.iter()) {
            var.set(shadow)?;
        }
        Ok(())
    }

    /// Returns reference to shadow weights.
    pub fn shadow_weights(&self) -> &[Tensor] {
        &self.shadow_weights
    }
}

// ============================================================================
// 4. Dataset & DataLoader
// ============================================================================

/// In-memory dataset holding feature tensors and optional label tensors.
pub struct TensorDataset {
    inputs: Tensor,
    targets: Option<Tensor>,
    length: usize,
}

impl TensorDataset {
    pub fn new(inputs: Tensor, targets: Option<Tensor>) -> Result<Self> {
        let length = inputs.dim(0)?;
        if let Some(t) = &targets {
            let t_len = t.dim(0)?;
            if length != t_len {
                return Err(candle_core::Error::ShapeMismatchBinaryOp {
                    lhs: inputs.shape().clone(),
                    rhs: t.shape().clone(),
                    op: "TensorDataset",
                });
            }
        }
        Ok(Self { inputs, targets, length })
    }

    pub fn len(&self) -> usize {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.length == 0
    }
}

/// Mini-batch iterator with deterministic or random index shuffling.
pub struct DataLoader<'a> {
    dataset: &'a TensorDataset,
    batch_size: usize,
    indices: Vec<usize>,
    current_idx: usize,
    drop_last: bool,
}

impl<'a> DataLoader<'a> {
    pub fn new(dataset: &'a TensorDataset, batch_size: usize, shuffle: bool, drop_last: bool) -> Self {
        Self::new_seeded(dataset, batch_size, shuffle, drop_last, 42)
    }

    /// Mini-batch iterator with explicit seed for deterministic shuffling across platforms.
    pub fn new_seeded(
        dataset: &'a TensorDataset,
        batch_size: usize,
        shuffle: bool,
        drop_last: bool,
        seed: u64,
    ) -> Self {
        let length = dataset.len();
        let mut indices: Vec<usize> = (0..length).collect();
        if shuffle {
            use rand::SeedableRng;
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            indices.shuffle(&mut rng);
        }
        Self {
            dataset,
            batch_size: batch_size.max(1),
            indices,
            current_idx: 0,
            drop_last,
        }
    }
}

impl<'a> Iterator for DataLoader<'a> {
    type Item = Result<(Tensor, Option<Tensor>)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.indices.len() {
            return None;
        }

        let remaining = self.indices.len() - self.current_idx;
        if remaining < self.batch_size && self.drop_last {
            return None;
        }

        let cur_batch_size = remaining.min(self.batch_size);
        let batch_indices: Vec<u32> = self.indices[self.current_idx..self.current_idx + cur_batch_size]
            .iter()
            .map(|&idx| idx as u32)
            .collect();

        self.current_idx += cur_batch_size;

        let device = self.dataset.inputs.device();
        let idx_tensor = match Tensor::from_vec(batch_indices, (cur_batch_size,), device) {
            Ok(t) => t,
            Err(e) => return Some(Err(e)),
        };

        let batch_inputs = match self.dataset.inputs.index_select(&idx_tensor, 0) {
            Ok(t) => t,
            Err(e) => return Some(Err(e)),
        };

        let batch_targets = match &self.dataset.targets {
            Some(t) => match t.index_select(&idx_tensor, 0) {
                Ok(sel) => Some(sel),
                Err(e) => return Some(Err(e)),
            },
            None => None,
        };

        Some(Ok((batch_inputs, batch_targets)))
    }
}

// ============================================================================
// 5. Training History & Telemetry Logger
// ============================================================================

/// Records step-by-step and epoch-by-epoch training metrics.
#[derive(Debug, Clone)]
pub struct TrainingHistory {
    pub steps: Vec<usize>,
    pub loss: Vec<f32>,
    pub lr: Vec<f64>,
    pub epoch_loss: Vec<f32>,
    #[cfg(not(target_arch = "wasm32"))]
    start_time: Instant,
}

impl Default for TrainingHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl TrainingHistory {
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            loss: Vec::new(),
            lr: Vec::new(),
            epoch_loss: Vec::new(),
            #[cfg(not(target_arch = "wasm32"))]
            start_time: Instant::now(),
        }
    }

    /// Logs a single step's metrics.
    pub fn log_step(&mut self, step: usize, loss: f32, lr: f64) {
        self.steps.push(step);
        self.loss.push(loss);
        self.lr.push(lr);
    }

    /// Logs epoch aggregate loss.
    pub fn log_epoch(&mut self, epoch_loss: f32) {
        self.epoch_loss.push(epoch_loss);
    }

    /// Returns elapsed training time in seconds.
    pub fn elapsed_seconds(&self) -> f64 {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.start_time.elapsed().as_secs_f64()
        }
        #[cfg(target_arch = "wasm32")]
        {
            0.0
        }
    }

    /// Returns the most recent loss value.
    pub fn latest_loss(&self) -> Option<f32> {
        self.loss.last().copied()
    }
}
