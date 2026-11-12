//! Universal Golden Parity Harness, Precision Diagnostics, and Drift Verification.
//!
//! **Stability: Experimental** — API subject to change
//!
//! Provides automated numerical verification ensuring bit-level agreement between
//! Candle (Rust), PyTorch (Python), and WebAssembly inference:
//! - [`GoldenParityHarness`]: Fluent evaluation engine checking max absolute error, RMSE, SQNR, and cosine similarity.
//! - [`ParityReport`]: Detailed numerical telemetry identifying exact index and magnitude of floating-point drift.
//! - [`ParityError`]: Structured errors raised when numerical divergence violates tolerances.

use crate::quantization::{compute_cosine_similarity, compute_mae_rmse, compute_sqnr};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Numerical error raised during golden parity validation.
#[derive(Debug, Error, Clone, PartialEq, Serialize, Deserialize)]
pub enum ParityError {
    #[error("Tensor dimension mismatch: expected {expected} elements, got {actual}")]
    ShapeMismatch { expected: usize, actual: usize },

    #[error("Non-finite float value encountered at index {index}: {value}")]
    NonFiniteEncountered { index: usize, value: f32 },

    #[error(
        "Parity tolerance exceeded for '{tensor_name}': max error {max_abs_error:.6e} > tol {tolerance:.6e} (SQNR: {sqnr_db:.1} dB)"
    )]
    ToleranceExceeded {
        tensor_name: String,
        max_abs_error: f32,
        tolerance: f32,
        sqnr_db: f32,
        cosine_similarity: f32,
        max_error_index: usize,
    },
}

/// Comprehensive numerical precision and parity report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParityReport {
    pub tensor_name: String,
    pub count: usize,
    pub max_abs_error: f32,
    pub mean_abs_error: f32,
    pub rmse: f32,
    pub sqnr_db: f32,
    pub cosine_similarity: f32,
    pub max_error_index: usize,
    pub tolerance: f32,
    pub passed: bool,
}

/// Builder for [`GoldenParityHarness`].
#[derive(Debug, Clone)]
pub struct GoldenParityHarnessBuilder {
    tolerance: f32,
    min_sqnr_db: f32,
    min_cosine: f32,
}

impl GoldenParityHarnessBuilder {
    pub fn tolerance(mut self, tol: f32) -> Self {
        self.tolerance = tol;
        self
    }

    pub fn min_sqnr_db(mut self, sqnr: f32) -> Self {
        self.min_sqnr_db = sqnr;
        self
    }

    pub fn min_cosine(mut self, cosine: f32) -> Self {
        self.min_cosine = cosine;
        self
    }

    pub fn build(self) -> GoldenParityHarness {
        GoldenParityHarness {
            tolerance: self.tolerance,
            min_sqnr_db: self.min_sqnr_db,
            min_cosine: self.min_cosine,
        }
    }
}

/// Universal parity verification harness.
///
/// # Examples
///
/// ```rust
/// use spodeian_ml_utils::GoldenParityHarness;
///
/// let harness = GoldenParityHarness::new(1e-4);
/// let candidate = vec![1.00001, 2.00002, 3.00001];
/// let reference = vec![1.0, 2.0, 3.0];
///
/// let report = harness.verify("layer1.weight", &candidate, &reference).unwrap();
/// assert!(report.passed);
/// assert!(report.max_abs_error < 1e-4);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoldenParityHarness {
    pub tolerance: f32,
    pub min_sqnr_db: f32,
    pub min_cosine: f32,
}

impl Default for GoldenParityHarness {
    fn default() -> Self {
        Self {
            tolerance: 1e-4,
            min_sqnr_db: 45.0, // Minimum 45 dB SQNR
            min_cosine: 0.9999,
        }
    }
}

impl GoldenParityHarness {
    /// Creates a new parity harness with the specified maximum absolute tolerance.
    pub fn new(tolerance: f32) -> Self {
        Self {
            tolerance,
            ..Default::default()
        }
    }

    /// Creates a fluent builder for configuring tolerances and quality metrics.
    pub fn builder() -> GoldenParityHarnessBuilder {
        GoldenParityHarnessBuilder {
            tolerance: 1e-4,
            min_sqnr_db: 45.0,
            min_cosine: 0.9999,
        }
    }

    /// Verifies candidate output against a reference golden slice.
    pub fn verify(
        &self,
        tensor_name: &str,
        candidate: &[f32],
        reference: &[f32],
    ) -> Result<ParityReport, ParityError> {
        if candidate.len() != reference.len() {
            return Err(ParityError::ShapeMismatch {
                expected: reference.len(),
                actual: candidate.len(),
            });
        }

        let mut max_err = 0.0f32;
        let mut max_idx = 0;

        for (i, (&c, &r)) in candidate.iter().zip(reference.iter()).enumerate() {
            if !c.is_finite() {
                return Err(ParityError::NonFiniteEncountered { index: i, value: c });
            }
            if !r.is_finite() {
                return Err(ParityError::NonFiniteEncountered { index: i, value: r });
            }

            let diff = (c - r).abs();
            if diff > max_err {
                max_err = diff;
                max_idx = i;
            }
        }

        let (mae, rmse) = compute_mae_rmse(candidate, reference);
        let sqnr_db = compute_sqnr(candidate, reference);
        let cosine_similarity = compute_cosine_similarity(candidate, reference);

        let passed = max_err <= self.tolerance
            && sqnr_db >= self.min_sqnr_db
            && cosine_similarity >= self.min_cosine;

        let report = ParityReport {
            tensor_name: tensor_name.to_string(),
            count: candidate.len(),
            max_abs_error: max_err,
            mean_abs_error: mae,
            rmse,
            sqnr_db,
            cosine_similarity,
            max_error_index: max_idx,
            tolerance: self.tolerance,
            passed,
        };

        if !passed {
            return Err(ParityError::ToleranceExceeded {
                tensor_name: tensor_name.to_string(),
                max_abs_error: max_err,
                tolerance: self.tolerance,
                sqnr_db,
                cosine_similarity,
                max_error_index: max_idx,
            });
        }

        Ok(report)
    }

    /// Asserts that candidate matches reference, panicking with a detailed diagnostic report on failure.
    pub fn assert_parity(&self, tensor_name: &str, candidate: &[f32], reference: &[f32]) {
        match self.verify(tensor_name, candidate, reference) {
            Ok(report) => {
                assert!(
                    report.passed,
                    "Parity check failed for '{}': {:?}",
                    tensor_name, report
                );
            }
            Err(e) => {
                panic!("Golden parity validation failed for '{tensor_name}': {e}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parity_harness_pass() {
        let harness = GoldenParityHarness::new(1e-3);
        let reference = vec![1.0, 2.0, 3.0, 4.0];
        let candidate = vec![1.0001, 1.9999, 3.0002, 3.9998];

        let report = harness
            .verify("layer1.output", &candidate, &reference)
            .unwrap();
        assert!(report.passed);
        assert!(report.max_abs_error < 1e-3);
        assert!(report.cosine_similarity > 0.9999);
        assert!(report.sqnr_db > 60.0);
    }

    #[test]
    fn test_parity_harness_failure_detection() {
        let harness = GoldenParityHarness::new(1e-4);
        let reference = vec![1.0, 2.0, 3.0, 4.0];
        let candidate = vec![1.0, 2.0, 3.05, 4.0]; // Drift at index 2

        let err = harness
            .verify("layer1.output", &candidate, &reference)
            .unwrap_err();
        match err {
            ParityError::ToleranceExceeded {
                max_error_index,
                max_abs_error,
                ..
            } => {
                assert_eq!(max_error_index, 2);
                assert!((max_abs_error - 0.05).abs() < 1e-5);
            }
            other => panic!("Expected ToleranceExceeded, got {:?}", other),
        }
    }
}
