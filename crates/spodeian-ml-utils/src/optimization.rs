//! Quasi-Newton Limited-Memory BFGS (L-BFGS) Optimization & Curvature Tracking Engine.
//!
//! **Stability: Experimental** — API subject to change
//!
//! Provides second-order optimization and inverse Hessian-vector products without storing full $d \times d$ matrices:
//! - [`LbfgsCurvatureTracker`]: Online L-BFGS two-loop recursion evaluating $H_k v$ in $O(m \cdot d)$ time.
//! - [`LbfgsOptimizer`]: Pure Rust second-order optimizer with backtracking Armijo line search.
//! - [`OptimizationResult`]: Detailed convergence telemetry for nonlinear estimation and portfolio risk parity.

use serde::{Deserialize, Serialize};

/// Online Limited-Memory BFGS Curvature Tracker & Inverse Hessian Vector Product (iHVP) engine.
///
/// Evaluates $H_k v \approx (\nabla^2 f)^{-1} v$ via two-loop recursion using $m$ historical displacement pairs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LbfgsCurvatureTracker {
    history_size: usize,
    eps: f64,
    s_history: Vec<Vec<f64>>,
    y_history: Vec<Vec<f64>>,
    rho_history: Vec<f64>,
}

impl LbfgsCurvatureTracker {
    /// Creates a new L-BFGS curvature tracker with the specified memory depth $m$ (typically 5 to 20).
    pub fn new(history_size: usize) -> Self {
        Self {
            history_size: history_size.max(1),
            eps: 1e-10,
            s_history: Vec::with_capacity(history_size),
            y_history: Vec::with_capacity(history_size),
            rho_history: Vec::with_capacity(history_size),
        }
    }

    /// Resets all curvature history.
    pub fn reset(&mut self) {
        self.s_history.clear();
        self.y_history.clear();
        self.rho_history.clear();
    }

    /// Records a new parameter displacement $s = x_{k+1} - x_k$ and gradient difference $y = g_{k+1} - g_k$.
    ///
    /// Enforces the curvature condition $y^T s > 0$ to guarantee positive definiteness.
    pub fn update(&mut self, s: Vec<f64>, y: Vec<f64>) -> bool {
        assert_eq!(s.len(), y.len(), "Dimension mismatch between s and y");
        let ys: f64 = s.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum();

        if ys <= self.eps {
            // Curvature condition violated; skip update to preserve positive definiteness
            return false;
        }

        let rho = 1.0 / ys;
        if self.s_history.len() >= self.history_size {
            self.s_history.remove(0);
            self.y_history.remove(0);
            self.rho_history.remove(0);
        }

        self.s_history.push(s);
        self.y_history.push(y);
        self.rho_history.push(rho);
        true
    }

    /// Number of curvature pairs currently retained.
    pub fn len(&self) -> usize {
        self.s_history.len()
    }

    /// Returns `true` if no curvature history has been recorded yet.
    pub fn is_empty(&self) -> bool {
        self.s_history.is_empty()
    }

    /// Computes the quasi-Newton search direction $p = -H_k g$ using the L-BFGS two-loop recursion.
    pub fn compute_search_direction(&self, grad: &[f64]) -> Vec<f64> {
        let n = grad.len();
        let m = self.s_history.len();

        if m == 0 {
            // First step / no history fallback: steepest descent (-grad)
            return grad.iter().map(|&g| -g).collect();
        }

        let mut q = grad.to_vec();
        let mut alphas = vec![0.0f64; m];

        // First loop: backward from newest (m-1) to oldest (0)
        for i in (0..m).rev() {
            let s_i = &self.s_history[i];
            let y_i = &self.y_history[i];
            let rho_i = self.rho_history[i];

            let sq: f64 = s_i.iter().zip(q.iter()).map(|(&a, &b)| a * b).sum();
            let alpha = rho_i * sq;
            alphas[i] = alpha;

            for j in 0..n {
                q[j] -= alpha * y_i[j];
            }
        }

        // Initial Hessian approximation scaling gamma_k = (s_{k-1}^T y_{k-1}) / (y_{k-1}^T y_{k-1})
        let last_idx = m - 1;
        let s_last = &self.s_history[last_idx];
        let y_last = &self.y_history[last_idx];
        let sy: f64 = s_last.iter().zip(y_last.iter()).map(|(&a, &b)| a * b).sum();
        let yy: f64 = y_last.iter().map(|&b| b * b).sum();
        let gamma = if yy > 1e-12 { sy / yy } else { 1.0 };

        let mut r: Vec<f64> = q.into_iter().map(|qi| gamma * qi).collect();

        // Second loop: forward from oldest (0) to newest (m-1)
        for i in 0..m {
            let s_i = &self.s_history[i];
            let y_i = &self.y_history[i];
            let rho_i = self.rho_history[i];

            let yr: f64 = y_i.iter().zip(r.iter()).map(|(&a, &b)| a * b).sum();
            let beta = rho_i * yr;
            let alpha_minus_beta = alphas[i] - beta;

            for j in 0..n {
                r[j] += alpha_minus_beta * s_i[j];
            }
        }

        // Return search direction p = -H_k g
        r.into_iter().map(|ri| -ri).collect()
    }
}

/// Result of an L-BFGS optimization run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub solution: Vec<f64>,
    pub final_loss: f64,
    pub iterations: usize,
    pub gradient_norm: f64,
    pub converged: bool,
}

/// Limited-Memory BFGS nonlinear numerical optimizer with Armijo backtracking line search.
///
/// # Examples
///
/// ```rust
/// use spodeian_ml_utils::LbfgsOptimizer;
///
/// // Minimize quadratic: f(x, y) = (x - 3)^2 + 2*(y + 1)^2
/// let optimizer = LbfgsOptimizer::new(50, 1e-6);
/// let res = optimizer.minimize(&[0.0, 0.0], |x| {
///     let loss = (x[0] - 3.0).powi(2) + 2.0 * (x[1] + 1.0).powi(2);
///     let grad = vec![2.0 * (x[0] - 3.0), 4.0 * (x[1] + 1.0)];
///     (loss, grad)
/// });
///
/// assert!(res.converged);
/// assert!((res.solution[0] - 3.0).abs() < 1e-4);
/// assert!((res.solution[1] - (-1.0)).abs() < 1e-4);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LbfgsOptimizer {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub history_size: usize,
    pub armijo_c1: f64,
    pub backtrack_rho: f64,
}

impl Default for LbfgsOptimizer {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            tolerance: 1e-6,
            history_size: 10,
            armijo_c1: 1e-4,
            backtrack_rho: 0.5,
        }
    }
}

impl LbfgsOptimizer {
    /// Creates a new L-BFGS optimizer with the specified maximum iterations and gradient norm tolerance.
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        Self {
            max_iterations,
            tolerance,
            ..Default::default()
        }
    }

    /// Minimizes an objective function `obj_and_grad` starting from initial parameter vector `x0`.
    ///
    /// The closure takes candidate parameters `&[f64]` and returns `(loss, gradient_vector)`.
    pub fn minimize<F>(&self, x0: &[f64], mut obj_and_grad: F) -> OptimizationResult
    where
        F: FnMut(&[f64]) -> (f64, Vec<f64>),
    {
        let mut x = x0.to_vec();
        let (mut loss, mut grad) = obj_and_grad(&x);
        let mut tracker = LbfgsCurvatureTracker::new(self.history_size);
        let mut iter = 0;

        while iter < self.max_iterations {
            let grad_norm = (grad.iter().map(|&g| g * g).sum::<f64>()).sqrt();
            if grad_norm <= self.tolerance {
                return OptimizationResult {
                    solution: x,
                    final_loss: loss,
                    iterations: iter,
                    gradient_norm: grad_norm,
                    converged: true,
                };
            }

            // Quasi-Newton search direction p = -H_k g
            let p = tracker.compute_search_direction(&grad);
            let dir_dot_grad: f64 = p.iter().zip(grad.iter()).map(|(&a, &b)| a * b).sum();

            // Guard: ensure p is a descent direction
            let p = if dir_dot_grad >= 0.0 {
                grad.iter().map(|&g| -g).collect::<Vec<f64>>()
            } else {
                p
            };
            let dir_dot_grad: f64 = p.iter().zip(grad.iter()).map(|(&a, &b)| a * b).sum();

            // Backtracking Armijo line search
            let mut alpha = if iter == 0 {
                (1.0 / grad_norm).min(1.0)
            } else {
                1.0f64
            };
            let mut new_x = vec![0.0f64; x.len()];
            let mut new_loss;
            let mut new_grad;

            loop {
                for i in 0..x.len() {
                    new_x[i] = x[i] + alpha * p[i];
                }
                let eval = obj_and_grad(&new_x);
                new_loss = eval.0;
                new_grad = eval.1;

                if new_loss <= loss + self.armijo_c1 * alpha * dir_dot_grad || alpha < 1e-12 {
                    break;
                }
                alpha *= self.backtrack_rho;
            }

            // Displacement and gradient difference
            let s: Vec<f64> = new_x.iter().zip(x.iter()).map(|(&nx, &ox)| nx - ox).collect();
            let y: Vec<f64> = new_grad.iter().zip(grad.iter()).map(|(&ng, &og)| ng - og).collect();
            tracker.update(s, y);

            x = new_x;
            loss = new_loss;
            grad = new_grad;
            iter += 1;
        }

        let grad_norm = (grad.iter().map(|&g| g * g).sum::<f64>()).sqrt();
        OptimizationResult {
            solution: x,
            final_loss: loss,
            iterations: iter,
            gradient_norm: grad_norm,
            converged: grad_norm <= self.tolerance,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quadratic_minimization() {
        // Minimize f(x, y) = (x - 3)^2 + 2*(y + 1)^2
        // Minimum at (3.0, -1.0) with loss = 0
        let optimizer = LbfgsOptimizer::new(50, 1e-6);
        let res = optimizer.minimize(&[0.0, 0.0], |x| {
            let loss = (x[0] - 3.0).powi(2) + 2.0 * (x[1] + 1.0).powi(2);
            let grad = vec![2.0 * (x[0] - 3.0), 4.0 * (x[1] + 1.0)];
            (loss, grad)
        });

        assert!(res.converged);
        assert!((res.solution[0] - 3.0).abs() < 1e-4);
        assert!((res.solution[1] - (-1.0)).abs() < 1e-4);
        assert!(res.final_loss < 1e-7);
    }

    #[test]
    fn test_rosenbrock_minimization() {
        // Standard Rosenbrock banana function: f(x, y) = (1 - x)^2 + 100*(y - x^2)^2
        // Global minimum at (1.0, 1.0) with loss = 0
        let optimizer = LbfgsOptimizer::new(150, 1e-5);
        let res = optimizer.minimize(&[-1.2, 1.0], |p| {
            let x = p[0];
            let y = p[1];
            let loss = (1.0 - x).powi(2) + 100.0 * (y - x * x).powi(2);
            let gx = -2.0 * (1.0 - x) - 400.0 * x * (y - x * x);
            let gy = 200.0 * (y - x * x);
            (loss, vec![gx, gy])
        });

        assert!(res.converged || res.final_loss < 1e-4);
        assert!((res.solution[0] - 1.0).abs() < 0.05);
        assert!((res.solution[1] - 1.0).abs() < 0.05);
    }
}
