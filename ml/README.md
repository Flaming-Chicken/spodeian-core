# ai-systems-rs: Systems-Level Rust ML & Hardware Compute Kernels

[![Rust Toolchain](https://img.shields.io/badge/Rust-1.80%2B-orange.svg)](Cargo.toml)
[![Candle Core](https://img.shields.io/badge/Candle-0.8.2-blue.svg)](crates/ai-candle)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](LICENSE)

High-performance, zero-Python, memory-safe Rust machine learning architectures and hardware compute kernels, covering 263 topics across 18 categories.

## Architecture & Crates

This repository is organized as a Cargo workspace with four core crates and two source tracks:

- **`crates/ai-candle` (`candle/`)**:
  Zero-Python, zero-GC implementations using Hugging Face's [Candle](https://github.com/huggingface/candle) framework across all 18 categories:
  - `00_foundations` through `17_deployment_low_level_and_capstones`
  - High-performance tensor operations, layer architectures, optimizers, and loss functions.

- **`crates/ai-kernel` (`kernels/`)**:
  Hardware compute kernel track providing:
  - Native CPU SIMD / vectorized kernel algorithms.
  - WebGPU (WGSL) compute shader pipelines with automatic CPU fallback.

- **`crates/ai-architect-schema`**:
  Shared Rust `serde` data models, architecture descriptors, and specification envelopes.

- **`crates/aix`**:
  Repository toolchain CLI: contract checks, crate synchronisation, code scaffolding, and verification.

## Building & Testing

```bash
# Check all workspace crates
cargo check --workspace

# Run tests across crates
cargo test -p ai-candle
cargo test -p ai-kernel

# Build release binaries
cargo build --release --workspace
```
