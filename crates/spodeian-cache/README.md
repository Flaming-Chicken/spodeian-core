# `spodeian-cache`

[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](../../LICENSE)
[![Status](https://img.shields.io/badge/stability-stable%20(v1.1)-success.svg)]()

**Preferential multi-tier storage routing and bounded in-memory caching for web, desktop, and mobile.**

---

## Overview

`spodeian-cache` provides preferential routing and tiered memory caching to prevent out-of-memory (OOM) conditions during long-running sessions, high-frequency factor calculation, and real-time audio synthesis:

- **In-Memory Tier (`BoundedMemoryCache`)**: Sub-microsecond RAM cache with amortized $O(1)$ LRU eviction and access generation counters. Ideal for 60–120 FPS UI loops and hot working sets.
- **Synchronous Tiered Store (`TieredMemoryStore`)**: Spills evicted items to cryptographic Content-Addressed Storage (CAS) on disk. Its fallback index persists across restarts; `flush()` saves the in-memory tier too.
- **Asynchronous Tiered Store (`AsyncTieredStore`)**: Write-through: every `put` is durable in IndexedDB (web) or CAS (desktop) before it returns, with a bounded in-memory working set in front.
- **Preferential Storage Router**: `PreferentialRouter` decides the tier; `PreferentialStore` carries it out: immutable or large binaries to the Cache API, structured records to IndexedDB, small non-JSON values to LocalStorage, and named CAS references on desktop.
- **Browser Backends (`web`, wasm32 only)**: `IndexedDbStore`, `CacheApiStore`, `LocalStorageStore`, and `StorageManager` helpers (`request_persistent_storage`, `is_storage_persisted`, `storage_estimate`).
- **Named References**: `ContentAddressedStorage::put_named` and `get_named` find data by a stable key after a restart.

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

### Durable app state on any platform

```rust,ignore
use spodeian_cache::PreferentialStore;

#[cfg(target_arch = "wasm32")]
let store = PreferentialStore::open_web("my-app").await?;
#[cfg(not(target_arch = "wasm32"))]
let store = PreferentialStore::open_native(app_data_dir)?;

let tier = store.save_json("state", &state).await?; // IndexedDb on the web, NativeCas on desktop
let restored: Option<AppState> = store.load_json("state").await?;
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
