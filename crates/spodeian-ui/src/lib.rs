//! Universal egui components and application shell for Spodeian projects.
//!
//! Provides responsive ScreenConstraints, WCAG 2.1 AAA accessible theming, and layout primitives.

pub mod layout;
pub mod theme;

pub use layout::ScreenConstraints;
pub use theme::{ThemeMode, apply_theme};
