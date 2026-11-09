//! Universal Model Formats & Conversion Subsystem (`sml-runtime::format`).
//!
//! Provides bidirectional converters and loaders for:
//! - GGUF v2/v3 (`.gguf`)
//! - Legacy GGML binary (`.bin`, `ggml`, `ggmf`, `ggjt`)
//! - SafeTensors (`.safetensors`)

pub mod converter;
pub mod ggml;
pub mod gguf;

pub use converter::*;
pub use ggml::*;
pub use gguf::*;
