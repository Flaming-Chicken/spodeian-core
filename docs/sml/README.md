# sml (in spodeian-core): High-Performance Systems-Level Rust ML & Hardware Compute Kernels

[![Rust Toolchain](https://img.shields.io/badge/Rust-1.85%2B%20%7C%202024%20Edition-orange.svg)](Cargo.toml)
[![Candle Core](https://img.shields.io/badge/Candle-0.11-blue.svg)](../../crates/sml-candle)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](LICENSE)

High-performance, zero-Python, memory-safe Rust machine learning runtime, local embedded columnar Parquet data lake, PyO3 bindings, and WebGPU/SIMD hardware compute kernels.

## Workspace Architecture & Member Crates

| Crate | Directory | Purpose |
| :--- | :--- | :--- |
| **`sml-runtime`** | [`crates/sml-runtime`](../../crates/sml-runtime) | Production inference & training runtime: SafeTensors streaming loader, continuous flow ODE/SDE solvers (Adaptive RK45, Symplectic Verlet, PCEC Adams-Moulton 4, Self-Optimizing Meta-Integrator), universal quantization (12 precisions including Log8/DistributionFloat4/Posit), and advanced sampling (Mirostat v2, Min-P, XTC, Dynamic Temp). |
| **`sml-lake`** | [`crates/sml-lake`](../../crates/sml-lake) | Embedded columnar local data lake: Apache Parquet partition persistence, debounced write-ahead log (WAL), LRU staleness eviction, and streaming Arrow record batches. |
| **`sml-py-bridge`** | [`crates/sml-py-bridge`](../../crates/sml-py-bridge) | Zero-copy PyO3 Python FFI bridge exposing Rust data transformations, fast numerical solvers, and Arrow arrays to `spodeian-ml-py`. |
| **`sml-candle`** | [`crates/sml-candle`](../../crates/sml-candle) | Pure Rust neural network implementations using Hugging Face's Candle framework covering foundations, continuous flow models, and Mamba-2 state space heads. |
| **`sml-kernel`** | [`crates/sml-kernel`](../../crates/sml-kernel) | Hardware compute acceleration: WebGPU (WGSL) compute pipelines with automatic CPU SIMD vectorized fallbacks. |
| **`sml-schema`** | [`crates/sml-schema`](../../crates/sml-schema) | Strongly typed Rust `serde` descriptors, architecture blueprints (`AiDesignSpec`), and golden vector validation envelopes. |
| **`sml-tools`** | [`crates/sml-tools`](../../crates/sml-tools) | CLI toolchain: cross-language parity verifiers, notebook scaffolding, model quantization inspection, and test harness execution. |

## Quick Verification

```bash
# Check all workspace crates
cargo check --workspace

# Run core runtime tests
cargo test -p sml-runtime

# Execute data lake and Parquet benchmarks
cargo test -p sml-lake
```
