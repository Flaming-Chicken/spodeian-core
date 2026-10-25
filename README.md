# Spodeian Core

[![CI](https://github.com/Flaming-Chicken/spodeian-core/actions/workflows/ci.yml/badge.svg)](https://github.com/Flaming-Chicken/spodeian-core/actions)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)

**Cross-Project Shared Ecosystem Foundation: Preferential Multi-Tier Caching, WCAG 2.1 AAA Design Tokens, Telemetry, Data Lake, Numerical Stability, and PyO3 Native Interop**

---

## Member Crates

| Crate | Stability | Description |
| :--- | :--- | :--- |
| [`spodeian-cache`](file:///crates/spodeian-cache) | **Stable (v1.0)** | Preferential CAS and multi-tier storage routing immutable blobs to Cache API / Native CAS, structured records to IndexedDB, and bounded in-memory caching (`BoundedMemoryCache`, `TieredMemoryStore`) with persistent fallback to prevent RAM overfilling. |
| [`spodeian-tokens`](file:///crates/spodeian-tokens) | **Stable (v1.0)** | WCAG 2.1 AAA themes, colorblind-safe palettes (Deuteranopia, Protanopia, Tritanopia, Monochrome), spectrogram colormaps, and CLI accessibility. |
| [`spodeian-telemetry`](file:///crates/spodeian-telemetry) | **Stable (v1.0)** | Universal zero-overhead tracing, circular in-memory `LogRingBuffer` for UI consoles, and crash report panic hooks. |
| [`spodeian-export`](file:///crates/spodeian-export) | **Stable (v1.0)** | Fast streaming serialization supporting CSV, JSON, and compressed BSON formats. |
| [`spodeian-ui`](file:///crates/spodeian-ui) | **Stable (v1.0)** | Universal egui shell components: responsive `ScreenConstraints`, accessible focus styling, reactive frame pacer, and `LogConsoleWidget`. |
| [`spodeian-web-utils`](file:///crates/spodeian-web-utils) | **Stable (v1.0)** | Cross-target web and native runtime detection, clipboard copy, and file download fallbacks. |
| [`spodeian-lake`](file:///crates/spodeian-lake) | **Beta (v0.1)** | Local embedded columnar Parquet data lake, debounced WAL persistence, partition streaming, and retention eviction. |
| [`spodeian-ml-utils`](file:///crates/spodeian-ml-utils) | **Beta (v0.1)** | Numerical stability (Higham nearest PSD projection, safe float sanitization, iteration limiter), universal quantization (12 precisions including Log8), multi-horizon heads, and RLS solvers. |
| [`spodeian-py-bridge`](file:///crates/spodeian-py-bridge) | **Beta (v0.1)** | High-throughput zero-copy Python bridge via PyO3, Apache Arrow arrays, and rolling statistical indicators. |

---

## Storage & Memory Architecture

`spodeian-cache` provides preferential routing and tiered memory caching:

```
┌────────────────────────────────────────────────────────┐
│               In-Memory Tier (RAM)                     │
│  - Format: Native Rust structs (HashMap + LRU queue)   │
│  - Latency: Sub-microsecond (< 1 µs)                   │
│  - Bounded Capacity: Prevents browser & desktop OOM    │
└──────────────────────────┬─────────────────────────────┘
                           │ (Eviction on Capacity Threshold)
                           ▼
┌────────────────────────────────────────────────────────┐
│          Persistent Fallback Tier (IndexedDB / CAS)    │
│  - Format: Structured Documents (Web) / CAS Blobs (OS) │
│  - Capacity: Multi-megabyte to gigabyte quota          │
│  - Role: Prevents memory overfill while retaining state│
└────────────────────────────────────────────────────────┘
```

---

## Testing & Quality Assurance

This repository utilizes [`cargo-nextest`](https://nexte.st) as its standard test runner for high-throughput, parallelized test execution with retry profiling and JUnit reporting.

```bash
# Run all workspace unit and integration tests with nextest
cargo nextest run --workspace --exclude spodeian-py-bridge

# Run CI profile with retry and fail-fast policies
cargo nextest run --workspace --exclude spodeian-py-bridge --profile ci

# Check code formatting
cargo fmt --all -- --check

# Check strict clippy linter rules
cargo clippy --workspace --exclude spodeian-py-bridge --all-targets -- -D warnings
```

---

## Engineering Tenets

- **Zero Compiler Warnings**: Enforced across native and WASM via `-D warnings`.
- **Zero Clippy Warnings**: Strict linting across all targets and unit tests.
- **100% Test Coverage**: All tests isolated in dedicated integration test suites under `tests/`.
- **Universal Accessibility**: Strict adherence to WCAG 2.1 AAA contrast and colorblind safety guidelines.
