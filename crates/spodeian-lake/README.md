# `spodeian-lake`

[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](../../LICENSE)
[![Status](https://img.shields.io/badge/stability-beta%20(v0.1)-yellow.svg)]()

**Embedded columnar Apache Parquet data lake, Arrow streaming, debounced WAL, and synthetic domain generators.**

---

## Overview

`spodeian-lake` provides an embedded columnar storage engine for high-throughput machine learning datasets, backtesting ticks, and audio telemetry:

- **Multi-Batch Parquet Engine (`TabularDataset`)**: Consumes all Arrow record batches without data truncation, converting between typed Rust records and compressed Parquet files.
- **Write-Ahead Log (`DebouncedWalWriter`)**: Fast in-memory buffer with periodic background disk flushes, guaranteeing data durability without stalling ingestion pipelines.
- **Procedural Domain Data Generators (`SyntheticMarketGenerator`, `SyntheticAcousticGenerator`)**: In-memory synthetic simulation engines generating realistic Geometric Brownian Motion price ticks and audio telemetry chirps without external fixture files.
- **Tabular Schema Bridge (`TabularSchema`)**: Universal bidirectional conversion trait mapping strongly typed domain records to columnar tables.

---

## Quickstart

```rust
use spodeian_lake::{SyntheticMarketGenerator, TabularSchema, FinancialObservation};

// Procedural generation of 10 market ticks
let generator = SyntheticMarketGenerator::new(42);
let ticks = generator.generate_ticks("NVDA", 10, 120.0, 0.35);

// Convert to columnar TabularDataset
let dataset = FinancialObservation::to_tabular_dataset(&ticks).unwrap();
assert_eq!(dataset.row_count, 10);
```
