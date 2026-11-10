//! # Spodeian Core
//!
//! **Stability: Stable** — semver-guaranteed since v1.0
//!
//! The central ecosystem umbrella facade re-exporting all Spodeian micro-crates:
//! - `parallel`: Zero-overhead Rayon and WASM concurrency facade (`spodeian-parallel`).
//! - `telemetry`: Zero-cost tracing subscriber, panic hooks, and in-memory log buffer (`spodeian-telemetry`).
//! - `tokens`: WCAG AAA design tokens, KaTeX math symbols, and color presets (`spodeian-tokens`).
//! - `export`: Universal serialization (JSON, CSV, Bincode, SafeTensors, BSON) (`spodeian-export`).
//! - `cache`: Preferential multi-tier caching (Web Cache API, IndexedDB, CAS) (`spodeian-cache`).
//! - `web_utils`: WebAssembly DOM helpers, clipboard, and blob transfers (`spodeian-web-utils`).
//! - `ui`: Reusable `egui` components, responsive layout constraints, and diagnostic viewers (`spodeian-ui`).
//! - `lake`: Local Apache Parquet data lake, Arrow streaming, and metadata registry (`spodeian-lake`).
//! - `ml_utils`: Candle neural inference, SafeTensors loader, RLS online adaptation, and dynamic heads (`spodeian-ml-utils`).
//!
//! # Example
//!
//! ```rust
//! use spodeian_core::prelude::*;
//!
//! // Online sliding statistics with O(1) constant-time updates
//! let mut stats = OnlineSlidingStats::new(5);
//! stats.push(10.0);
//! stats.push(12.0);
//! stats.push(11.0);
//! assert_eq!(stats.len(), 3);
//! assert!((stats.mean() - 11.0).abs() < 1e-6);
//! ```

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
    pub use spodeian_telemetry::{
        LogEntry, LogRingBuffer, global_ring_buffer, init_default, init_telemetry,
    };

    #[cfg(feature = "tokens")]
    pub use spodeian_tokens::*;

    #[cfg(feature = "ui")]
    pub use spodeian_ui::ScreenConstraints;

    #[cfg(feature = "lake")]
    pub use spodeian_lake::{
        SyntheticAcousticGenerator, SyntheticMarketGenerator, TabularDataset, TabularSchema,
    };

    #[cfg(feature = "ml-utils")]
    pub use spodeian_ml_utils::{
        ComputeDevice, ComputeRouter, ContinuousIntegrator, EulerStepper, ExponentialMovingAverage,
        GoldenParityHarness, HeunPredictorCorrector, LbfgsCurvatureTracker, LbfgsOptimizer,
        OnlineSlidingCovariance, OnlineSlidingStats, ParityReport, Rk4Stepper, RollingWindow,
        StepResult,
    };
}
