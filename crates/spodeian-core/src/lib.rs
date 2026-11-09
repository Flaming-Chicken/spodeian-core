//! # Spodeian Core
//!
//! **Stability: Stable** — semver-guaranteed since v1.0
//!
//! The central ecosystem umbrella facade re-exporting all Spodeian micro-crates:
//! - [`parallel`]: Zero-overhead Rayon and WASM concurrency facade ([`spodeian_parallel`]).
//! - [`telemetry`]: Zero-cost tracing subscriber, panic hooks, and in-memory log buffer ([`spodeian_telemetry`]).
//! - [`tokens`]: WCAG AAA design tokens, KaTeX math symbols, and color presets ([`spodeian_tokens`]).
//! - [`export`]: Universal serialization (JSON, CSV, Bincode, SafeTensors, BSON) ([`spodeian_export`]).
//! - [`cache`]: Preferential multi-tier caching (Web Cache API, IndexedDB, CAS) ([`spodeian_cache`]).
//! - [`web_utils`]: WebAssembly DOM helpers, clipboard, and blob transfers ([`spodeian_web_utils`]).
//! - [`ui`]: Reusable `egui` components, responsive layout constraints, and diagnostic viewers ([`spodeian_ui`]).
//! - [`lake`]: Local Apache Parquet data lake, Arrow streaming, and metadata registry ([`spodeian_lake`]).
//! - [`ml_utils`]: Candle neural inference, SafeTensors loader, RLS online adaptation, and dynamic heads ([`spodeian_ml_utils`]).

#[cfg(feature = "parallel")]
pub use spodeian_parallel as parallel;

#[cfg(feature = "telemetry")]
pub use spodeian_telemetry as telemetry;

#[cfg(feature = "tokens")]
pub use spodeian_tokens as tokens;

#[cfg(feature = "export")]
pub use spodeian_export as export;

#[cfg(feature = "cache")]
pub use spodeian_cache as cache;

#[cfg(feature = "web-utils")]
pub use spodeian_web_utils as web_utils;

#[cfg(feature = "ui")]
pub use spodeian_ui as ui;

#[cfg(feature = "lake")]
pub use spodeian_lake as lake;

#[cfg(feature = "ml-utils")]
pub use spodeian_ml_utils as ml_utils;

/// Standard prelude re-exporting common ecosystem traits and types.
pub mod prelude {
    #[cfg(feature = "parallel")]
    pub use spodeian_parallel::prelude::*;

    #[cfg(feature = "telemetry")]
    pub use spodeian_telemetry::{LogEntry, LogRingBuffer, global_ring_buffer, init_default, init_telemetry};

    #[cfg(feature = "tokens")]
    pub use spodeian_tokens::*;

    #[cfg(feature = "ui")]
    pub use spodeian_ui::ScreenConstraints;

    #[cfg(feature = "ml-utils")]
    pub use spodeian_ml_utils::{
        ContinuousIntegrator, EulerStepper, HeunPredictorCorrector, Rk4Stepper, StepResult,
    };
}
