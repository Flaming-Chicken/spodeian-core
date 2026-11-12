//! PyTorch 2 Parity Learning Rate Schedulers for `sml-runtime`.
//!
//! Provides deterministic, zero-Python learning rate schedules:
//! - CosineAnnealingLR
//! - LinearWarmupCosineAnnealing
//! - OneCycleLR (Leslie Smith 1cycle policy)
//! - ExponentialLR
//! - StepLR & MultiStepLR
//! - ReduceLROnPlateau

use crate::optim::Optimizer;
use std::f64::consts::PI;

/// Trait defining learning rate scheduler contract.
pub trait LRScheduler {
    /// Steps the scheduler forward by one epoch/iteration and updates the optimizer's learning rate.
    fn step(&mut self, optimizer: &mut dyn Optimizer);

    /// Computes the learning rate for the current step.
    fn get_lr(&self) -> f64;

    /// Returns the current step/epoch index.
    fn current_step(&self) -> usize;
}

// ============================================================================
// 1. CosineAnnealingLR
// ============================================================================

pub struct CosineAnnealingLR {
    base_lr: f64,
    t_max: usize,
    eta_min: f64,
    current_step: usize,
}

impl CosineAnnealingLR {
    pub fn new(base_lr: f64, t_max: usize, eta_min: f64) -> Self {
        Self {
            base_lr,
            t_max: t_max.max(1),
            eta_min,
            current_step: 0,
        }
    }
}

impl LRScheduler for CosineAnnealingLR {
    fn step(&mut self, optimizer: &mut dyn Optimizer) {
        self.current_step += 1;
        let lr = self.get_lr();
        optimizer.set_learning_rate(lr);
    }

    fn get_lr(&self) -> f64 {
        if self.current_step >= self.t_max {
            self.eta_min
        } else {
            let progress = self.current_step as f64 / self.t_max as f64;
            self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1.0 + (PI * progress).cos())
        }
    }

    fn current_step(&self) -> usize {
        self.current_step
    }
}

// ============================================================================
// 2. LinearWarmupCosineAnnealing
// ============================================================================

pub struct LinearWarmupCosineAnnealing {
    base_lr: f64,
    warmup_steps: usize,
    total_steps: usize,
    eta_min: f64,
    current_step: usize,
}

impl LinearWarmupCosineAnnealing {
    pub fn new(base_lr: f64, warmup_steps: usize, total_steps: usize, eta_min: f64) -> Self {
        Self {
            base_lr,
            warmup_steps,
            total_steps: total_steps.max(warmup_steps + 1),
            eta_min,
            current_step: 0,
        }
    }
}

impl LRScheduler for LinearWarmupCosineAnnealing {
    fn step(&mut self, optimizer: &mut dyn Optimizer) {
        self.current_step += 1;
        let lr = self.get_lr();
        optimizer.set_learning_rate(lr);
    }

    fn get_lr(&self) -> f64 {
        if self.current_step < self.warmup_steps {
            // Linear warmup: [0, base_lr]
            self.base_lr * (self.current_step as f64 / self.warmup_steps.max(1) as f64)
        } else if self.current_step >= self.total_steps {
            self.eta_min
        } else {
            // Cosine decay
            let decay_steps = self.total_steps - self.warmup_steps;
            let current_decay = self.current_step - self.warmup_steps;
            let progress = current_decay as f64 / decay_steps as f64;
            self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1.0 + (PI * progress).cos())
        }
    }

    fn current_step(&self) -> usize {
        self.current_step
    }
}

// ============================================================================
// 3. OneCycleLR (Leslie Smith 1cycle Policy)
// ============================================================================

pub struct OneCycleLR {
    max_lr: f64,
    total_steps: usize,
    pct_start: f64,
    div_factor: f64,
    final_div_factor: f64,
    current_step: usize,
}

impl OneCycleLR {
    pub fn new(max_lr: f64, total_steps: usize, pct_start: f64) -> Self {
        Self {
            max_lr,
            total_steps: total_steps.max(1),
            pct_start: pct_start.clamp(0.01, 0.99),
            div_factor: 25.0,
            final_div_factor: 1e4,
            current_step: 0,
        }
    }
}

impl LRScheduler for OneCycleLR {
    fn step(&mut self, optimizer: &mut dyn Optimizer) {
        self.current_step += 1;
        let lr = self.get_lr();
        optimizer.set_learning_rate(lr);
    }

    fn get_lr(&self) -> f64 {
        let initial_lr = self.max_lr / self.div_factor;
        let final_lr = initial_lr / self.final_div_factor;
        let step_up = (self.total_steps as f64 * self.pct_start) as usize;
        let step_down = self.total_steps.saturating_sub(step_up);

        if self.current_step <= step_up {
            // Phase 1: Cosine annealing up to max_lr
            let progress = self.current_step as f64 / step_up.max(1) as f64;
            initial_lr + 0.5 * (self.max_lr - initial_lr) * (1.0 - (PI * progress).cos())
        } else if self.current_step <= self.total_steps {
            // Phase 2: Cosine annealing down to final_lr
            let progress = (self.current_step - step_up) as f64 / step_down.max(1) as f64;
            final_lr + 0.5 * (self.max_lr - final_lr) * (1.0 + (PI * progress).cos())
        } else {
            final_lr
        }
    }

    fn current_step(&self) -> usize {
        self.current_step
    }
}

// ============================================================================
// 4. ExponentialLR
// ============================================================================

pub struct ExponentialLR {
    initial_lr: f64,
    gamma: f64,
    current_step: usize,
}

impl ExponentialLR {
    pub fn new(initial_lr: f64, gamma: f64) -> Self {
        Self {
            initial_lr,
            gamma,
            current_step: 0,
        }
    }
}

impl LRScheduler for ExponentialLR {
    fn step(&mut self, optimizer: &mut dyn Optimizer) {
        self.current_step += 1;
        let lr = self.get_lr();
        optimizer.set_learning_rate(lr);
    }

    fn get_lr(&self) -> f64 {
        self.initial_lr * self.gamma.powi(self.current_step as i32)
    }

    fn current_step(&self) -> usize {
        self.current_step
    }
}

// ============================================================================
// 5. StepLR & MultiStepLR
// ============================================================================

pub struct StepLR {
    initial_lr: f64,
    step_size: usize,
    gamma: f64,
    current_step: usize,
}

impl StepLR {
    pub fn new(initial_lr: f64, step_size: usize, gamma: f64) -> Self {
        Self {
            initial_lr,
            step_size: step_size.max(1),
            gamma,
            current_step: 0,
        }
    }
}

impl LRScheduler for StepLR {
    fn step(&mut self, optimizer: &mut dyn Optimizer) {
        self.current_step += 1;
        let lr = self.get_lr();
        optimizer.set_learning_rate(lr);
    }

    fn get_lr(&self) -> f64 {
        let count = self.current_step / self.step_size;
        self.initial_lr * self.gamma.powi(count as i32)
    }

    fn current_step(&self) -> usize {
        self.current_step
    }
}

pub struct MultiStepLR {
    initial_lr: f64,
    milestones: Vec<usize>,
    gamma: f64,
    current_step: usize,
}

impl MultiStepLR {
    pub fn new(initial_lr: f64, mut milestones: Vec<usize>, gamma: f64) -> Self {
        milestones.sort_unstable();
        Self {
            initial_lr,
            milestones,
            gamma,
            current_step: 0,
        }
    }
}

impl LRScheduler for MultiStepLR {
    fn step(&mut self, optimizer: &mut dyn Optimizer) {
        self.current_step += 1;
        let lr = self.get_lr();
        optimizer.set_learning_rate(lr);
    }

    fn get_lr(&self) -> f64 {
        let count = self.milestones.iter().filter(|&&m| self.current_step >= m).count();
        self.initial_lr * self.gamma.powi(count as i32)
    }

    fn current_step(&self) -> usize {
        self.current_step
    }
}

// ============================================================================
// 6. ReduceLROnPlateau
// ============================================================================

pub struct ReduceLROnPlateau {
    current_lr: f64,
    factor: f64,
    patience: usize,
    min_lr: f64,
    best_metric: f64,
    bad_epochs: usize,
    current_step: usize,
    minimize: bool,
}

impl ReduceLROnPlateau {
    pub fn new(initial_lr: f64, factor: f64, patience: usize, min_lr: f64, minimize: bool) -> Self {
        Self {
            current_lr: initial_lr,
            factor: factor.clamp(0.01, 0.99),
            patience,
            min_lr,
            best_metric: if minimize { f64::INFINITY } else { f64::NEG_INFINITY },
            bad_epochs: 0,
            current_step: 0,
            minimize,
        }
    }

    pub fn step_metric(&mut self, metric: f64, optimizer: &mut dyn Optimizer) {
        self.current_step += 1;
        let improved = if self.minimize {
            metric < self.best_metric
        } else {
            metric > self.best_metric
        };

        if improved {
            self.best_metric = metric;
            self.bad_epochs = 0;
        } else {
            self.bad_epochs += 1;
            if self.bad_epochs > self.patience {
                self.current_lr = (self.current_lr * self.factor).max(self.min_lr);
                optimizer.set_learning_rate(self.current_lr);
                self.bad_epochs = 0;
            }
        }
    }
}
