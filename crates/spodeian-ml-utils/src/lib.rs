//! Spodeian ML Utils
//!
//! SafeTensors streaming loader, dynamic multi-horizon/quantile heads, RLS online adaptation,
//! and dual-speed TTA (Test-Time Adaptation) for Candle neural networks across the Spodeian ecosystem.

pub mod adaptation;
pub mod heads;
pub mod loader;

pub use adaptation::*;
pub use heads::*;
pub use loader::*;
