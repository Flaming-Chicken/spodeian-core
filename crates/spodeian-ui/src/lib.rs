//! Universal egui components and application shell for Spodeian projects.
//!
//! **Stability: Stable** — semver-guaranteed since v1.0
//!
//! Provides responsive ScreenConstraints, WCAG 2.1 AAA accessible theming, and layout primitives.

pub mod layout;
pub mod theme;

#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
pub use layout::ScreenConstraints;
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
pub use theme::{ThemeMode, apply_theme};
