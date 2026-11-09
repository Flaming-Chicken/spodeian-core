# `spodeian-tokens`

[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](../../LICENSE)
[![Status](https://img.shields.io/badge/stability-stable%20(v1.0)-success.svg)]()

**WCAG 2.1 AAA Design Tokens, Colorblind-Safe Palettes, and Scientific Colormaps.**

---

## Overview

`spodeian-tokens` defines visual constants and accessibility standards across both `egui` (desktop/native) and Leptos (WASM/web):

- **WCAG 2.1 AAA Contrast**: Pre-calibrated 7:1+ contrast ratios for Dark, Light, HighContrastDark, and HighContrastLight themes.
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
