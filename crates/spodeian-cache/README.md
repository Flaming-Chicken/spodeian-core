# `spodeian-cache`

[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](../../LICENSE)
[![Status](https://img.shields.io/badge/stability-stable%20(v1.0)-success.svg)]()

**Preferential multi-tier storage routing and bounded in-memory caching for web, desktop, and mobile.**

---

## Overview

`spodeian-cache` provides preferential routing and tiered memory caching to prevent out-of-memory (OOM) conditions during long-running sessions, high-frequency factor calculation, and real-time audio synthesis:

- **In-Memory Tier (`BoundedMemoryCache`)**: Sub-microsecond RAM cache with amortized $O(1)$ LRU eviction and access generation counters. Ideal for 60–120 FPS UI loops and hot working sets.
- **Persistent Fallback Tier (`TieredMemoryStore`)**: Automatic spillover to IndexedDB (on Web) or cryptographic Content-Addressed Storage (CAS) on desktop/server, ensuring zero data loss when RAM limits are reached.
- **Preferential Storage Router**: Routes immutable assets (`.safetensors`, `.parquet`, `.wasm`) to Cache API/CAS, structured records to IndexedDB, and settings to LocalStorage.

---

## Quickstart

```rust
use spodeian_cache::BoundedMemoryCache;

// Create an in-memory cache bounded to 1,000 items
let mut cache = BoundedMemoryCache::new(1000);

// Insert key-value pairs
cache.put("factor_momentum_1d", vec![0.12, -0.05, 0.33]);

// Fast sub-microsecond retrieval
if let Some(factors) = cache.get(&"factor_momentum_1d") {
    println!("Loaded {} factors", factors.len());
}
```

---

## Architecture

```
┌────────────────────────────────────────────────────────┐
│               In-Memory Tier (RAM)                     │
│  - Format: Native Rust structs (HashMap + LRU queue)   │
│  - Latency: Sub-microsecond (< 1 µs)                   │
│  - Capacity: Configurable bounded limit                │
└──────────────────────────┬─────────────────────────────┘
                           │ (Eviction on Capacity Threshold)
                           ▼
┌────────────────────────────────────────────────────────┐
│          Persistent Fallback Tier (IndexedDB / CAS)    │
│  - Format: Structured Documents (Web) / CAS Blobs (OS) │
│  - Capacity: Multi-megabyte to gigabyte quota          │
│  - Role: Retains state safely without process crashes  │
└────────────────────────────────────────────────────────┘
```
