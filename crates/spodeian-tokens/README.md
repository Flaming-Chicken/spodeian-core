# `spodeian-tokens`

[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](../../LICENSE)
[![Status](https://img.shields.io/badge/stability-stable%20(v1.1)-success.svg)]()

**WCAG 2.1 AAA Design Tokens, Colorblind-Safe Palettes, and Scientific Colormaps.**

---

## Overview

`spodeian-tokens` defines visual constants and accessibility standards across both `egui` (desktop/native) and Leptos (WASM/web):

- **WCAG 2.1 Contrast**: Every theme's text and status colours meet AA (4.5:1) on its background and surface, and the high-contrast themes' text meets AAA (7:1); tests enforce both. `border` is a decorative divider and must not be a control's only outline.
- **CSS Custom Properties**: `theme_stylesheet(prefix)` emits every theme's palette, selected by `[data-theme="..."]`, so web stylesheets never duplicate colour values. `MotionPreference` and `FontScale` emit motion and font-scale properties.
- **Vision Impairment Compensation**: Calibrated candlestick and curve palettes for Deuteranopia (green-weak), Protanopia (red-weak), Tritanopia (blue-yellow), and Monochrome.
- **Acoustic Spectrogram Colormaps**: Perceptually uniform color gradients for audio frequency spectrograms.
- **Responsive Screen Breakpoints**: Standardized device breakpoints (UltraTight, Tight, Mobile, Desktop).

---

## Quickstart

```rust
use spodeian_tokens::{ThemeMode, ColorblindMode, CandlestickColors};

// Cycle theme modes
let theme = ThemeMode::Dark;
let next_theme = theme.next();

// Vision-safe financial candlestick colors
let palette = CandlestickColors::for_mode(ColorblindMode::Deuteranopia);
println!("Up candle RGB: {:?}", palette.up);
```
