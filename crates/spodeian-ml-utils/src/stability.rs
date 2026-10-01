//! Numerical stability, float sanitization, and matrix regularization utilities.
//!
//! **Stability: Experimental** — API subject to change
//!
//! Provides traits and algorithms for preventing $\text{NaN}$ / $\pm\infty$ contamination,
//! regularizing ill-conditioned covariance matrices via Higham (2002) nearest PSD projection,
//! and enforcing strict convergence bounds on iterative numerical solvers.

use thiserror::Error;

/// Error returned when iterative numerical procedures fail to converge or violate constraints.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum ConvergenceError {
    /// Maximum iteration limit reached without meeting tolerance.
    #[error("Maximum iterations exceeded: reached {iterations} iterations with residual {residual:.6e} > tol {tolerance:.6e}")]
    MaxIterationsExceeded {
        /// Number of iterations executed.
        iterations: usize,
        /// Last observed residual.
        residual: f64,
        /// Target tolerance.
        tolerance: f64,
    },
    /// Numerical divergence detected (NaN or infinity encountered).
    #[error("Divergence detected at iteration {iteration}: residual is non-finite ({residual})")]
    Divergence {
        /// Iteration index at failure.
        iteration: usize,
        /// Non-finite residual value.
        residual: f64,
    },
    /// Matrix dimensions are invalid or non-square.
    #[error("Invalid matrix dimensions: expected square matrix of size {expected}x{expected}, got {rows}x{cols}")]
    InvalidDimensions {
        /// Expected dimension.
        expected: usize,
        /// Actual rows.
        rows: usize,
        /// Actual columns.
        cols: usize,
    },
}

/// Trait for inspecting and sanitizing floating-point slices and vectors.
pub trait SanitizeFloats {
    /// Replaces non-finite values (NaN, +inf, -inf) in-place with `fallback`.
    ///
    /// Returns the number of values replaced.
    fn replace_non_finite(&mut self, fallback: f64) -> usize;

    /// Checks if all values in the container are finite.
    fn is_finite_vector(&self) -> bool;

    /// Clamps all values in-place to the closed interval `[min, max]`.
    fn clamp_range(&mut self, min: f64, max: f64);
}

impl SanitizeFloats for [f64] {
    #[inline]
    fn replace_non_finite(&mut self, fallback: f64) -> usize {
        let mut count = 0;
        for x in self.iter_mut() {
            if !x.is_finite() {
                *x = fallback;
                count += 1;
            }
        }
        count
    }

    #[inline]
    fn is_finite_vector(&self) -> bool {
        self.iter().all(|x| x.is_finite())
    }

    #[inline]
    fn clamp_range(&mut self, min: f64, max: f64) {
        for x in self.iter_mut() {
            if x.is_nan() {
                *x = min;
            } else {
                *x = x.clamp(min, max);
            }
        }
    }
}

impl SanitizeFloats for Vec<f64> {
    #[inline]
    fn replace_non_finite(&mut self, fallback: f64) -> usize {
        self.as_mut_slice().replace_non_finite(fallback)
    }

    #[inline]
    fn is_finite_vector(&self) -> bool {
        self.as_slice().is_finite_vector()
    }

    #[inline]
    fn clamp_range(&mut self, min: f64, max: f64) {
        self.as_mut_slice().clamp_range(min, max);
    }
}

impl SanitizeFloats for [f32] {
    #[inline]
    fn replace_non_finite(&mut self, fallback: f64) -> usize {
        let fallback_f32 = fallback as f32;
        let mut count = 0;
        for x in self.iter_mut() {
            if !x.is_finite() {
                *x = fallback_f32;
                count += 1;
            }
        }
        count
    }

    #[inline]
    fn is_finite_vector(&self) -> bool {
        self.iter().all(|x| x.is_finite())
    }

    #[inline]
    fn clamp_range(&mut self, min: f64, max: f64) {
        let min_f32 = min as f32;
        let max_f32 = max as f32;
        for x in self.iter_mut() {
            if x.is_nan() {
                *x = min_f32;
            } else {
                *x = x.clamp(min_f32, max_f32);
            }
        }
    }
}

impl SanitizeFloats for Vec<f32> {
    #[inline]
    fn replace_non_finite(&mut self, fallback: f64) -> usize {
        self.as_mut_slice().replace_non_finite(fallback)
    }

    #[inline]
    fn is_finite_vector(&self) -> bool {
        self.as_slice().is_finite_vector()
    }

    #[inline]
    fn clamp_range(&mut self, min: f64, max: f64) {
        self.as_mut_slice().clamp_range(min, max);
    }
}

/// Robust division that returns `fallback` when the denominator is zero, subnormal, or NaN.
#[inline]
pub fn safe_div(numerator: f64, denominator: f64, fallback: f64) -> f64 {
    if !numerator.is_finite() || !denominator.is_finite() || denominator.abs() < 1e-15 {
        fallback
    } else {
        numerator / denominator
    }
}

/// Robust f32 division that returns `fallback` when the denominator is zero, subnormal, or NaN.
#[inline]
pub fn safe_div_f32(numerator: f32, denominator: f32, fallback: f32) -> f32 {
    if !numerator.is_finite() || !denominator.is_finite() || denominator.abs() < 1e-7 {
        fallback
    } else {
        numerator / denominator
    }
}

/// Symmetrizes an $n \times n$ matrix represented as a flat row-major slice.
///
/// Computes $A \gets \frac{1}{2}(A + A^T)$.
pub fn symmetrize_matrix(matrix: &mut [f64], n: usize) -> Result<(), ConvergenceError> {
    if matrix.len() != n * n {
        return Err(ConvergenceError::InvalidDimensions {
            expected: n,
            rows: matrix.len() / n.max(1),
            cols: n,
        });
    }

    for i in 0..n {
        for j in (i + 1)..n {
            let avg = 0.5 * (matrix[i * n + j] + matrix[j * n + i]);
            matrix[i * n + j] = avg;
            matrix[j * n + i] = avg;
        }
    }
    Ok(())
}

/// Projects a symmetric $n \times n$ matrix $A$ to the nearest Positive Semi-Definite (PSD) matrix.
///
/// Implements spectral eigenvalue truncation with optional ridge floor:
/// $$A_{\text{psd}} = V \max(\Lambda, \epsilon I) V^T$$
/// where $A = V \Lambda V^T$ via symmetric Jacobi diagonalization, followed by symmetrization.
///
/// # Arguments
/// * `matrix` - Flat row-major array of size $n \times n$.
/// * `n` - Matrix dimension.
/// * `min_eigenvalue` - Small positive ridge eigenvalue $\epsilon \ge 0$ (e.g. `1e-8`).
///
/// # Returns
/// Projected PSD matrix as a flat `Vec<f64>` of length $n^2$.
pub fn nearest_psd_matrix(
    matrix: &[f64],
    n: usize,
    min_eigenvalue: f64,
) -> Result<Vec<f64>, ConvergenceError> {
    if matrix.len() != n * n {
        return Err(ConvergenceError::InvalidDimensions {
            expected: n,
            rows: matrix.len() / n.max(1),
            cols: n,
        });
    }

    if n == 0 {
        return Ok(Vec::new());
    }

    if n == 1 {
        let val = matrix[0].max(min_eigenvalue);
        return Ok(vec![val]);
    }

    // Work on a mutable copy
    let mut a = matrix.to_vec();
    symmetrize_matrix(&mut a, n)?;

    // Eigenvectors V initialized to Identity
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    // Classical Jacobi eigenvalue algorithm for symmetric matrix
    let max_sweeps = 50;
    let eps = 1e-12;

    for _sweep in 0..max_sweeps {
        let mut max_offdiag = 0.0f64;
        for i in 0..n {
            for j in (i + 1)..n {
                let off = a[i * n + j].abs();
                if off > max_offdiag {
                    max_offdiag = off;
                }
            }
        }

        if max_offdiag < eps {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                let apq = a[p * n + q];

                if apq.abs() < 1e-15 {
                    continue;
                }

                // Compute Jacobi rotation angle
                let theta = (aqq - app) / (2.0 * apq);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (theta * theta + 1.0).sqrt())
                } else {
                    -1.0 / (-theta + (theta * theta + 1.0).sqrt())
                };

                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                let tau = s / (1.0 + c);

                // Update diagonal elements
                a[p * n + p] = app - t * apq;
                a[q * n + q] = aqq + t * apq;
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;

                // Update off-diagonal elements
                for r in 0..n {
                    if r != p && r != q {
                        let arp = a[r * n + p];
                        let arq = a[r * n + q];
                        let new_arp = arp - s * (arq + tau * arp);
                        let new_arq = arq + s * (arp - tau * arq);
                        a[r * n + p] = new_arp;
                        a[p * n + r] = new_arp;
                        a[r * n + q] = new_arq;
                        a[q * n + r] = new_arq;
                    }
                }

                // Accumulate eigenvectors
                for r in 0..n {
                    let vrp = v[r * n + p];
                    let vrq = v[r * n + q];
                    v[r * n + p] = vrp - s * (vrq + tau * vrp);
                    v[r * n + q] = vrq + s * (vrp - tau * vrq);
                }
            }
        }
    }

    // Extract eigenvalues and floor them at min_eigenvalue
    let mut eigenvalues = vec![0.0; n];
    for i in 0..n {
        eigenvalues[i] = a[i * n + i].max(min_eigenvalue);
    }

    // Reconstruct A_psd = V * Lambda_+ * V^T
    let mut result = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += v[i * n + k] * eigenvalues[k] * v[j * n + k];
            }
            result[i * n + j] = sum;
        }
    }

    symmetrize_matrix(&mut result, n)?;
    Ok(result)
}

/// Tracks convergence of iterative numerical algorithms, preventing unbounded execution.
#[derive(Debug, Clone)]
pub struct IterationLimiter {
    /// Maximum allowed iterations.
    pub max_iterations: usize,
    /// Absolute residual convergence tolerance.
    pub tolerance: f64,
    /// Current iteration count.
    pub current_iteration: usize,
    /// Last observed residual.
    pub last_residual: f64,
}

impl IterationLimiter {
    /// Creates a new iteration limiter.
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        Self {
            max_iterations,
            tolerance,
            current_iteration: 0,
            last_residual: f64::INFINITY,
        }
    }

    /// Evaluates one step of an iterative process with the given residual.
    ///
    /// # Returns
    /// * `Ok(true)` if converged (residual $\le$ tolerance).
    /// * `Ok(false)` if continuing within iteration budget.
    /// * `Err(ConvergenceError)` if divergent or maximum iterations exceeded.
    pub fn step(&mut self, residual: f64) -> Result<bool, ConvergenceError> {
        if !residual.is_finite() {
            return Err(ConvergenceError::Divergence {
                iteration: self.current_iteration,
                residual,
            });
        }

        self.current_iteration += 1;
        self.last_residual = residual;

        if residual <= self.tolerance {
            Ok(true)
        } else if self.current_iteration >= self.max_iterations {
            Err(ConvergenceError::MaxIterationsExceeded {
                iterations: self.current_iteration,
                residual,
                tolerance: self.tolerance,
            })
        } else {
            Ok(false)
        }
    }

    /// Resets the counter for another run.
    pub fn reset(&mut self) {
        self.current_iteration = 0;
        self.last_residual = f64::INFINITY;
    }
}
