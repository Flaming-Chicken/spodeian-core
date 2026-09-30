//! Universal telemetry, logging, and panic capture system for Spodeian ecosystem.
//!
//! **Stability: Stable** — semver-guaranteed since v1.0
//!
//! Provides identical structured JSON or ANSI console logs on desktop and mobile,
//! and directs output to the browser developer console via `tracing-web` on WASM.

#[allow(unused_imports)]
use tracing::info;

/// Configures and initializes global tracing subscriber and panic handlers.
///
/// On native platforms (desktop, CLI, Android), installs `tracing_subscriber::fmt` with `EnvFilter`.
/// On WebAssembly (`wasm32`), installs `console_error_panic_hook` and `tracing-web`.
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
pub fn init_telemetry(default_filter: &str) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        use tracing_subscriber::{EnvFilter, fmt, prelude::*};
        let filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new(default_filter));

        let _ = tracing_subscriber::registry()
            .with(fmt::layer())
            .with(filter)
            .try_init();
    }

    #[cfg(target_arch = "wasm32")]
    {
        console_error_panic_hook::set_once();
        let _ = tracing_subscriber::fmt()
            .with_writer(tracing_web::MakeConsoleWriter)
            .without_time()
            .with_ansi(false)
            .try_init();
    }
}

/// Convenience initializer using standard desktop/egui filter:
/// `"info,wgpu=warn,egui=warn"`
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
pub fn init_default() {
    init_telemetry("info,wgpu=warn,egui=warn");
}
