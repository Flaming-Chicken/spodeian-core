# Spodeian Core

[![CI](https://github.com/Spodeian/spodeian-core/actions/workflows/ci.yml/badge.svg)](https://github.com/Spodeian/spodeian-core/actions)
[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-blue.svg)](LICENSE.md)

**Cross-Project Shared Ecosystem Foundation: Parallelism (`rayon`/WASM), WebGPU (`wgpu`), Multi-Tier Caching, WCAG 2.1 AAA Design Tokens, Telemetry, Interchange Export, Leptos/egui UI Primitives, Provenance, Statistical/Forecasting Kernels, and PyO3 Native Interop**

---

## Member Crates (19-Crate Workspace)

| Crate | Version | Role & Capabilities |
| :--- | :--- | :--- |
| [`spodeian-core`](crates/spodeian-core) | `^3.0.0` (`3.0.1` workspace) | Unified umbrella facade with template feature bundles (`preset-egui`, `preset-leptos-edge`, `preset-leptos-desktop`, `wgpu`, `kernel`, `schema`, `full`) and standard `prelude`. |
| [`spodeian-parallel`](crates/spodeian-parallel) | `1.0.0` | Zero-overhead concurrency facade executing **Rayon** (`par_iter`, `join`, `current_num_threads`) on native targets with transparent inline sequential fallback on `wasm32` / `no_std`. |
| [`spodeian-cache`](crates/spodeian-cache) | `1.1.0` | Preferential CAS and multi-tier storage routing immutable blobs to Web Cache API / Native CAS, structured documents to IndexedDB, and bounded in-memory caching (`BoundedMemoryCache`, `AsyncTieredStore`). |
| [`spodeian-persist`](crates/spodeian-persist) | `0.1.0` | Versioned `StateMigrator` upgrades (`schema_version`), atomic desktop snapshots with `.bak` recovery, and browser `StorageDiagnostics` helpers. |
| [`spodeian-tokens`](crates/spodeian-tokens) | `1.1.0` | WCAG 2.1 AAA themes (`ThemeMode`), colorblind-safe palettes (Deuteranopia, Protanopia, Tritanopia, Monochrome), colormaps, and contrast verification tokens. |
| [`spodeian-telemetry`](crates/spodeian-telemetry) | `1.1.0` | Zero-overhead `tracing` initialization across native & WASM (`tracing-web`), circular in-memory `LogRingBuffer`, panic hooks, and `TelemetryMetrics`. |
| [`spodeian-export`](crates/spodeian-export) | `2.0.0` | Streaming RFC 4180 CSV (incl. borrowed slice parser), JSON, Zlib-compressed BSON (`u64_i64_compat`), SafeTensors, PCM WAV (`16i`/`32f`), and `escape_html` for SVG/HTML reports. |
| [`spodeian-web-utils`](crates/spodeian-web-utils) | `1.1.0` | Cross-target runtime detection (`RuntimeTarget`, `is_tauri_desktop`, `is_wasm_browser`, `has_webgpu_support`, `hardware_concurrency`), async Tauri IPC (`invoke_tauri_ipc`), confirmed clipboard writes (`write_clipboard_text`), downloads, and `pwa` SSG post-processing (`inline_script_hashes`, `populate_service_worker`, `generate_sitemap_xml`, `generate_robots_txt`). |
| [`spodeian-leptos-ui`](crates/spodeian-leptos-ui) | `0.1.0` | Leptos & web DOM bridges (`normalize_theme`, `prepaint_theme_and_dyslexia_script`), `trap_tab_within_element` focus trap, `ErrorBoundaryRecovery`, `TelemetrySummary`, Schema.org `generate_json_ld_webapp`, and `detect_shortcut_conflicts`. |
| [`spodeian-ui`](crates/spodeian-ui) | `1.0.0` | Reusable `egui 0.36` shell primitives: `CommandPaletteWidget` + conflict detection, `UndoStack<T>`, `ThemePickerWidget`, `PerformanceMetricCard`, `LogConsoleWidget`, `ReactiveFramePacer`, `ScreenConstraints`, and `TelemetryDashboardModal`. |
| [`spodeian-provenance`](crates/spodeian-provenance) | `0.1.0` | Whole-token license verification, media provenance ledgers, and `ATTRIBUTIONS.txt` synthesis. |
| [`spodeian-stats`](crates/spodeian-stats) | `0.1.0` | Probability distributions, quantiles, OLS/WLS/Ridge/Huber/HAC regressions, stationarity tests, Hurst exponent, and cointegration primitives. |
| [`spodeian-forecast`](crates/spodeian-forecast) | `0.1.0` | Holt-Winters ETS, 1D Kalman state-space filter, GARCH(1,1) volatility, split-conformal prediction intervals, pinball loss, and Gaussian CRPS. |
| [`spodeian-audio`](crates/spodeian-audio) | `0.1.0` | Pure-Rust WAV PCM `16`/`24`/`32f` codec, linear resampling, acoustic quality metrics, and zero-allocation DSP kernels. |
| [`sml-kernel`](crates/sml-kernel) | `0.1.0` | WebGPU (`wgpu = "23"`) + WGSL compute pipeline cache (`WgpuContext::new_async`, `run_compute_shader`, `KernelDevice::summary`) with automatic CPU fallback. |
| [`sml-runtime`](crates/sml-runtime) | `0.2.0` | Candle inference, SafeTensors streaming loader, RK4/Euler continuous flow ODE integrators, L-BFGS optimizer, `GoldenParityHarness`, and `OnlineSlidingStats`. |
| [`sml-schema`](crates/sml-schema) | `0.1.0` | Typed `serde` models, tensor parameter descriptors, and golden verification vector manifests. |
| [`sml-lake`](crates/sml-lake) | `0.2.0` | Embedded Apache Parquet & Arrow columnar local data lake, LRU staleness eviction, WAL persistence, and synthetic market/acoustic generators. |
| [`sml-py-bridge`](crates/sml-py-bridge) | `0.1.0` | Desktop-only PyO3 FFI bridge exposing Rust augmentations, ODE solvers, and Arrow zero-copy buffers to Python. |

---

## Template Presets & Multi-Target Technology Support

All three official Spodeian application templates consume these crates flexibly according to their target architecture. See [`docs/TEMPLATE_INTEGRATION_GUIDE.md`](docs/TEMPLATE_INTEGRATION_GUIDE.md) for end-to-end recipes and architectural blueprints.

| Technology | `efficient-egui-template` | `efficient-leptos-template` | `efficient-leptos-edge-template` | `spodeian-core` Enabler |
|---|---|---|---|---|
| **UI Framework** | `egui 0.36` / `eframe 0.36` (`spodeian-ui`) | `Leptos 0.8` (`ssr` + `hydrate` + `csr`) | `Leptos 0.8` (`ssr` + `hydrate` only) | `spodeian-ui`, `spodeian-leptos-ui`, `spodeian-tokens` |
| **Desktop / Native** | Native Winit binary (`crates/desktop` + `mimalloc`) | Tauri v2 shell (`crates/desktop` + IPC commands) | *Excluded by design* (Zero desktop deps) | `spodeian-persist`, `spodeian-web-utils::invoke_tauri_ipc` |
| **Web / Edge** | Trunk CSR WASM SPA + PWA (`sw.js`, `_headers`) | `cargo-leptos` + `ssg` static HTML + WASM hydration | `cargo-leptos` + `ssg` + JSON-LD + `sitemap.xml` + `_redirects` | `spodeian-web-utils::pwa`, `spodeian-leptos-ui` |
| **Parallelism (`rayon`)** | Native Rayon pool / automatic WASM sequential fallback | Native SSR & Tauri Rayon pool / WASM fallback | Build-time SSR Rayon sweeps / WASM SIMD128 client | `spodeian-parallel` (`par_iter`, `join`, `current_num_threads`) |
| **WASM (`wasm32`)** | Full `wasm32-unknown-unknown` canvas + IDB + blobs | Dual `hydrate` (web) & `csr` (Tauri) WASM targets | Pure `hydrate` WASM bundle + strict hash CSP | `spodeian-cache`, `spodeian-web-utils`, `spodeian-parallel` |
| **WebGPU (`wgpu`)** | Primary `wgpu` renderer (`glow` fallback) + WGSL kernels | WebGPU compute probes + WGSL kernels via `sml-kernel` | `navigator.gpu` capability probe + WGSL shader ready | `sml-kernel` (`WgpuContext::new_async`, `run_compute_shader`) |

### Cargo Feature Presets on Umbrella Crate `spodeian-core`

```toml
# 1. Pure edge Leptos project (static SSR + WASM hydration, no egui or desktop deps)
spodeian-core = { git = "https://github.com/Spodeian/spodeian-core", default-features = false, features = ["preset-leptos-edge"] }

# 2. Hybrid Web + Tauri v2 Desktop Leptos project (add `wgpu` for GPU compute kernels)
spodeian-core = { git = "https://github.com/Spodeian/spodeian-core", default-features = false, features = ["preset-leptos-desktop", "wgpu"] }

# 3. Native + WASM Immediate-Mode egui project
spodeian-core = { git = "https://github.com/Spodeian/spodeian-core", default-features = false, features = ["preset-egui"] }
```

---

## Storage & Memory Architecture

`spodeian-cache` and `spodeian-persist` coordinate multi-tier storage across desktop and browser environments:

```
┌────────────────────────────────────────────────────────┐
│               In-Memory Tier (RAM)                     │
│  - Format: Native Rust structs (BoundedMemoryCache)    │
│  - Latency: Sub-microsecond (< 1 µs)                   │
│  - Bounded Capacity: Prevents browser & desktop OOM    │
└──────────────────────────┬─────────────────────────────┘
                           │ (Eviction / Sync Write-Through)
                           ▼
┌────────────────────────────────────────────────────────┐
│   Persistent Backing (IndexedDB / Cache API / Disk CAS)│
│  - Format: Versioned JSON/BSON + CAS Blobs             │
│  - Capacity: Multi-megabyte to gigabyte quota          │
│  - Migration: Schema upgrades via `StateMigrator`      │
└────────────────────────────────────────────────────────┘
```

---

## Testing & Quality Assurance

```bash
# Run workspace unit and integration tests (excluding desktop Python FFI bridge)
cargo test --workspace --exclude sml-py-bridge

# Or with cargo-nextest
cargo nextest run --workspace --all-targets --exclude sml-py-bridge --profile ci

# Verify formatting, strict clippy, and zero-warning documentation
cargo fmt --all -- --check
cargo clippy --workspace --exclude sml-py-bridge --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --workspace --exclude sml-py-bridge
```

---

## Engineering Tenets

- **Zero Compiler & Clippy Warnings**: Enforced across native and `wasm32` via `-D warnings` and `unsafe_code = "deny"`.
- **Zero-Warning Documentation**: Every public item is documented (`RUSTDOCFLAGS="-D warnings"`) with compile-tested doctests and KaTeX equations where applicable ([`docs/DOCUMENTATION_STANDARDS.md`](docs/DOCUMENTATION_STANDARDS.md)).
- **Universal Accessibility**: Strict WCAG 2.1 AAA / AA contrast validation, colorblind-safe palettes, dyslexia-friendly typography support, and keyboard focus management.

---

## License

Licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE.md). You may use,
change and share this software for any noncommercial purpose. For a
commercial license, contact [Spodeian](https://github.com/Spodeian). Earlier
releases keep the license they were published under.
