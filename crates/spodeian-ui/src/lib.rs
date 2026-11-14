//! Universal egui components and application shell for Spodeian projects.
//!
//! **Stability: Stable** — semver-guaranteed since v1.0
//!
//! Provides responsive ScreenConstraints, WCAG 2.1 AAA accessible theming, and layout primitives.

pub mod accessibility;
pub mod command_palette;
pub mod layout;
pub mod log_viewer;
pub mod pacing;
pub mod telemetry_modal;
pub mod theme;
pub mod undo;
pub mod widgets;

#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
pub use accessibility::{configure_accessible_focus, render_accessibility_panel};
#[doc = "**Stability: Experimental** — introduced in v3.0.0"]
pub use command_palette::{CommandEntry, CommandPaletteWidget};
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
pub use layout::ScreenConstraints;
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
pub use log_viewer::{LogConsoleWidget, LogLevelFilter};
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
pub use pacing::ReactiveFramePacer;
#[doc = "**Stability: Experimental** — introduced in v3.0.0"]
pub use telemetry_modal::{render_kpi_card, render_telemetry_dashboard_modal};
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
pub use theme::{ThemeMode, apply_theme};
#[doc = "**Stability: Experimental** — introduced in v3.0.0"]
pub use undo::UndoStack;
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
pub use widgets::{AccessibleSlider, PerformanceMetricCard, ThemePickerWidget};
