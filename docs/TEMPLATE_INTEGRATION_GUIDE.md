# Spodeian Template & Ecosystem Integration Guide

This guide explains how **`spodeian-core`** supports all three Spodeian application templates and how developers can combine **`tauri`**, **`egui`**, **`leptos`**, **`rayon` (`spodeian-parallel`)**, **`wasm32`**, and **`wgpu` (`sml-kernel`)** cleanly inside one workspace.

---

## 1. Choosing the Right Template

| Template | Primary Use Case | Crates Included | Deployment Targets |
|---|---|---|---|
| **`efficient-leptos-edge-template`** | Content sites, SEO landing pages, static dashboards, interactive calculators, zero-backend PWAs | `shared`, `app`, `web`, `ssg` | Cloudflare Pages, GitHub Pages, Netlify, AWS S3/CloudFront (Static SSR + WASM Hydration only) |
| **`efficient-leptos-template`** | Apps that ship **both** a pre-rendered website **and** a Tauri v2 desktop binary sharing one reactive DOM UI | `shared`, `app`, `web`, `ssg`, `desktop` | Cloudflare Pages / GitHub Pages (SSR+Hydration) **and** Windows/macOS/Linux desktop (`Tauri v2` CSR) |
| **`efficient-egui-template`** | Custom 2D/3D viewports, CAD/audio/science workstations, high-FPS plotting, immediate-mode canvases, single-binary GUI apps | `shared`, `app`, `desktop`, `web` | Native Desktop (`winit` + `wgpu`/`glow` + `mimalloc`), Static WASM SPA (Trunk), Android/iOS scaffolds |

---

## 2. Consuming `spodeian-core`

You can depend either on individual focused micro-crates (recommended for minimal compile times in `shared` and `web` crates) or on the `spodeian-core` umbrella crate with a preset bundle.

### Minimal Individual Dependencies (Used Inside Template Crates)
```toml
spodeian-parallel  = { git = "https://github.com/Spodeian/spodeian-core", tag = "v3.1.0" }
spodeian-tokens    = { git = "https://github.com/Spodeian/spodeian-core", tag = "v3.1.0" }
spodeian-export    = { git = "https://github.com/Spodeian/spodeian-core", tag = "v3.1.0" }
spodeian-persist   = { git = "https://github.com/Spodeian/spodeian-core", tag = "v3.1.0" }
spodeian-cache     = { git = "https://github.com/Spodeian/spodeian-core", tag = "v3.1.0" }
spodeian-telemetry = { git = "https://github.com/Spodeian/spodeian-core", tag = "v3.1.0" }
spodeian-web-utils = { git = "https://github.com/Spodeian/spodeian-core", tag = "v3.1.0" }
spodeian-leptos-ui = { git = "https://github.com/Spodeian/spodeian-core", tag = "v3.1.0" } # Leptos templates
spodeian-ui        = { git = "https://github.com/Spodeian/spodeian-core", tag = "v3.1.0" } # egui template
```

### Umbrella Presets (`crates/spodeian-core`)
```toml
spodeian-core = { git = "https://github.com/Spodeian/spodeian-core", default-features = false, features = ["preset-leptos-edge"] }
# Or: "preset-leptos-desktop", "preset-egui", optionally combined with "wgpu" / "kernel" / "lake" / "ml-utils"
```

---

## 3. Parallel Compute Across Native `rayon` and `wasm32` (`spodeian-parallel`)

WebAssembly browser builds running without SharedArrayBuffer / atomics cannot spawn native OS worker threads with standard Rayon. `spodeian-parallel` solves this with zero boilerplate:

```rust
use spodeian_parallel::prelude::*;

pub fn harmonic_batch(inputs: &[f32]) -> Vec<f32> {
    // Runs on Rayon's multi-threaded work-stealing pool on native / Tauri / SSR,
    // and compiles down to an inline sequential iterator on wasm32.
    let raw: Vec<f32> = inputs.par_iter().map(|&x| (x * 2.5).sin()).collect();

    // Introspect active concurrency anywhere:
    tracing::debug!(threads = current_num_threads(), parallel = is_hardware_parallel());
    raw
}
```

- **`crates/shared`** across all three templates enables `spodeian-parallel` so domain algorithms run multi-threaded inside native desktops, Tauri IPC handlers, and SSR generators while remaining 100% compatible with single-threaded `wasm32` browser targets and `--target-feature=+simd128` vectorization (configured in `.cargo/config.toml`).

---

## 4. WebGPU Graphics & WGSL Compute (`wgpu` + `sml-kernel`)

### Immediate-Mode Rendering (`efficient-egui-template`)
- `crates/app` and `crates/desktop` / `crates/web` enable `wgpu` by default (`eframe/wgpu`) with optional `glow` (`--no-default-features --features accesskit,persistence,glow`) for hardware lacking Vulkan/Metal/DX12/WebGPU.

### Portable WGSL Compute Shaders (`sml-kernel`)
Any native desktop binary, Tauri command, or async WASM task can dispatch WGSL compute shaders with cached pipelines and graceful CPU fallback via `sml-kernel`:

```rust
use sml_kernel::{KernelDevice, WgpuContext, wgpu_util::run_compute_shader};

pub async fn probe_or_run(shader_wgsl: &str, input: &[f32]) -> Vec<f32> {
    if let Some(ctx) = WgpuContext::new_async().await {
        let device = KernelDevice::Wgpu(ctx.clone());
        let summary = device.summary();
        tracing::info!("Active compute: {} (WGSL={})", summary.backend, summary.supports_wgsl_compute);
        // Dispatch via run_compute_shader(&ctx, shader_wgsl, "main", ...);
    }
    // Graceful CPU/Rayon fallback
    input.to_vec()
}
```

On browser WebAssembly (`wasm32`), use `spodeian_web_utils::has_webgpu_support()` to detect `navigator.gpu` synchronously before initiating async `WgpuContext::new_async().await` setup (avoid blocking on `pollster::block_on` in browsers).

---

## 5. Tauri v2 Desktop IPC Bridge (`efficient-leptos-template`)

In the hybrid Leptos template, heavy local compute (file ingestion, Rayon batch sweeps, Parquet/BSON processing, or `wgpu` shaders) belongs on the Rust backend in `crates/desktop/src/lib.rs`, exposed via `#[tauri::command]` and invoked cleanly from `crates/app/src/tauri_bridge.rs`:

1. Register the command in `crates/desktop/src/lib.rs`:
   ```rust
   #[tauri::command]
   pub fn compute_harmonic_sweep_native(/* args */) -> Vec<Vec<(f32, f32)>> { /* ... */ }
   ```
2. Detect the host at runtime and invoke safely from `crates/app` using `spodeian_web_utils::is_tauri_desktop()` and `spodeian_web_utils::invoke_tauri_ipc(command, args_json)`:
   - Inside Tauri desktop builds (`window.__TAURI_INTERNALS__` present), calls run natively via Rust.
   - When the same UI runs on the serverless web edge, `is_tauri_desktop()` returns `false` and execution cleanly stays in-browser via `shared` + WASM.

---

## 6. Pure Static Edge SSR + Hydration (`efficient-leptos-edge-template` & Web Target)

For zero-server static hosts (Cloudflare Pages, GitHub Pages, Netlify, S3/CloudFront):
1. `cargo-leptos build --release` builds:
   - `crates/ssg` with feature `ssr` (native binary)
   - `crates/web` with feature `hydrate` (WASM bundle hashed in `/pkg/`)
2. `ssg` renders every static route (`/` -> `index.html`, `/analytics` -> `analytics.html`, `/about` -> `about.html`, `/404` -> `404.html`), strips per-request nonces (`spodeian_web_utils::pwa::strip_nonce`), emits `sitemap.xml` and `robots.txt` (`generate_sitemap_xml` / `generate_robots_txt`), hashes inline scripts (`THEME_PREPAINT_SCRIPT`, Schema.org `generate_json_ld_webapp`) into `_headers` CSP directives, and stamps `sw.js` precache URLs + content hash build IDs.
3. Cloudflare `_headers` and `_redirects` guarantee immutable asset caching (`max-age=31536000, immutable` on `/pkg/*`), strict CSP with zero `unsafe-eval`, and clean 404 routing.
