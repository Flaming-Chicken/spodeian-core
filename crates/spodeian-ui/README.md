# `spodeian-ui`

[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](../../LICENSE)
[![Status](https://img.shields.io/badge/stability-stable%20(v1.0)-success.svg)]()

**Universal egui UI shell components, responsive layout constraints, and diagnostic viewers.**

---

## Overview

`spodeian-ui` provides standardized UI components built on `egui`:

- **Responsive Screen Constraints (`ScreenConstraints`)**: Dynamic layout classification (`is_mobile`, `is_mobile_portrait`, `is_ultra_tight`) for responsive dashboards.
- **Log Console Widget (`LogConsoleWidget`)**: High-performance scrolling diagnostic log viewer consuming `spodeian-telemetry::LogRingBuffer`.
- **Reactive Frame Pacer (`ReactiveFramePacer`)**: Dynamic FPS throttle dropping to low idle refresh when inputs are stationary, saving CPU and laptop battery.
- **Accessible Theme Manager**: Applies WCAG AAA dark/light styling and high-contrast outlines directly to `egui::Context`.

---

## Quickstart

```rust
use spodeian_ui::ScreenConstraints;

let constraints = ScreenConstraints::from_dimensions(600.0, 800.0);
if constraints.is_mobile {
    // Switch to single-column stacked layout
}
```
