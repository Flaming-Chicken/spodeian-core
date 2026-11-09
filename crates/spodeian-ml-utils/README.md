# `spodeian-ml-utils`

[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](../../LICENSE)
[![Status](https://img.shields.io/badge/stability-beta%20(v0.1)-yellow.svg)]()

**Universal Candle inference utils, online streaming statistics, hardware compute routing, numerical stability, ODE solvers, and L-BFGS optimizer.**

---

## Overview

`spodeian-ml-utils` provides the mathematical and machine learning substrate powering neural audio synthesis in `RainAI`, quantitative factor modeling in `Risky-Business`, continuous learning in `spodeian-ml`, and symbolic analysis in `algebra-engine`:

- **Universal Streaming Statistics (`streaming`)**: Constant-time $O(1)$ sliding window metrics (`RollingWindow`, `OnlineSlidingStats`, `ExponentialMovingAverage`, `OnlineSlidingCovariance`) eliminating $O(W \cdot N)$ repeated window scans.
- **Hardware & Compute Execution Router (`device`)**: Standardized hardware taxonomy (`ComputeDevice`) and capability discovery (`ComputeRouter`) bridging Candle, CUDA, Metal, WebGPU, and WASM SIMD.
- **Pure Rust Quasi-Newton L-BFGS (`optimization`)**: Two-loop recursion iHVP evaluation in $O(m \cdot d)$ time and backtracking Armijo line search with step-0 gradient scaling, delivering machine-epsilon precision.
- **Continuous ODE Solvers (`integrator`)**: 4th-Order Runge-Kutta (`Rk4Stepper`), Heun predictor-corrector, and Euler stepping with local truncation error estimation and adaptive step sizing.
- **Numerical Parity Verification (`parity`)**: `GoldenParityHarness` ensuring strict numerical equivalence between Candle inference and PyTorch reference models.
- **Numerical Stability & Nearest PSD Projection (`stability`)**: Higham modified Cholesky nearest PSD matrix projection and robust float sanitization.
- **Universal Quantized Inference Kernels (`quantization`, `logarithmic`)**: 12 quantization formats including NF4, INT8, Ternary 1.58-bit, BNN, and single-cycle L1-cached 8-bit logarithmic floats (`Log8`).

---

## Quickstart

```rust
use spodeian_ml_utils::{OnlineSlidingStats, LbfgsOptimizer};

// 1. O(1) Online sliding statistics
let mut stats = OnlineSlidingStats::new(10);
for price in [100.0, 101.5, 102.0, 99.8, 103.2] {
    stats.push(price);
}
println!("Sliding mean: {}, std: {}", stats.mean(), stats.std_dev());

// 2. Quasi-Newton second-order optimization
let optimizer = LbfgsOptimizer::new(50, 1e-6);
let res = optimizer.minimize(&[0.0, 0.0], |p| {
    let loss = (p[0] - 3.0).powi(2) + 2.0 * (p[1] + 1.0).powi(2);
    let grad = vec![2.0 * (p[0] - 3.0), 4.0 * (p[1] + 1.0)];
    (loss, grad)
});
assert!(res.converged);
```
