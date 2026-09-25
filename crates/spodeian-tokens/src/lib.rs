//! Ecosystem Design Tokens, WCAG 2.1 Accessibility & Typography (`spodeian-tokens`).
//!
//! Provides unified themes, color palettes, responsive screen breakpoints, and KaTeX rendering options
//! shared across Leptos (web) and egui (desktop/mobile) projects.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ThemeMode {
    #[default]
    Dark,
    Light,
    HighContrastDark,
    HighContrastLight,
}

impl ThemeMode {
    pub fn is_high_contrast(self) -> bool {
        matches!(self, Self::HighContrastDark | Self::HighContrastLight)
    }

    pub fn is_dark(self) -> bool {
        matches!(self, Self::Dark | Self::HighContrastDark)
    }
}

/// Standard 8-bit RGBA color representation with WCAG luminance calculations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ColorRgba {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl ColorRgba {
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }

    pub const fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Computes the relative luminance according to WCAG 2.1 specifications:
    /// $L = 0.2126 R + 0.7152 G + 0.0722 B$
    pub fn relative_luminance(self) -> f64 {
        fn channel_l(c: u8) -> f64 {
            let s = (c as f64) / 255.0;
            if s <= 0.04045 {
                s / 12.92
            } else {
                ((s + 0.055) / 1.055).powf(2.4)
            }
        }

        0.2126 * channel_l(self.r) + 0.7152 * channel_l(self.g) + 0.0722 * channel_l(self.b)
    }

    /// Computes the WCAG contrast ratio between two colors:
    /// $\text{Ratio} = \frac{L_1 + 0.05}{L_2 + 0.05}$
    pub fn contrast_ratio(c1: Self, c2: Self) -> f64 {
        let l1 = c1.relative_luminance();
        let l2 = c2.relative_luminance();
        let lighter = l1.max(l2);
        let darker = l1.min(l2);
        (lighter + 0.05) / (darker + 0.05)
    }

    /// Checks if contrast ratio meets WCAG 2.1 AA level for standard text ($\ge 4.5:1$).
    pub fn is_wcag_aa(c1: Self, c2: Self) -> bool {
        Self::contrast_ratio(c1, c2) >= 4.5
    }

    /// Checks if contrast ratio meets WCAG 2.1 AAA level for high contrast ($\ge 7.0:1$).
    pub fn is_wcag_aaa(c1: Self, c2: Self) -> bool {
        Self::contrast_ratio(c1, c2) >= 7.0
    }
}

/// Palette tokens for a UI theme.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThemePalette {
    pub background: ColorRgba,
    pub surface: ColorRgba,
    pub border: ColorRgba,
    pub text_primary: ColorRgba,
    pub text_secondary: ColorRgba,
    pub accent: ColorRgba,
    pub error: ColorRgba,
    pub success: ColorRgba,
}

impl ThemePalette {
    pub fn for_mode(mode: ThemeMode) -> Self {
        match mode {
            ThemeMode::Dark => Self {
                background: ColorRgba::rgb(18, 18, 18),
                surface: ColorRgba::rgb(28, 28, 28),
                border: ColorRgba::rgb(48, 48, 48),
                text_primary: ColorRgba::rgb(240, 240, 240),
                text_secondary: ColorRgba::rgb(160, 160, 160),
                accent: ColorRgba::rgb(90, 150, 255),
                error: ColorRgba::rgb(255, 85, 85),
                success: ColorRgba::rgb(80, 220, 120),
            },
            ThemeMode::Light => Self {
                background: ColorRgba::rgb(248, 249, 250),
                surface: ColorRgba::rgb(255, 255, 255),
                border: ColorRgba::rgb(220, 224, 230),
                text_primary: ColorRgba::rgb(30, 32, 36),
                text_secondary: ColorRgba::rgb(90, 95, 105),
                accent: ColorRgba::rgb(0, 100, 225),
                error: ColorRgba::rgb(215, 40, 40),
                success: ColorRgba::rgb(30, 160, 70),
            },
            ThemeMode::HighContrastDark => Self {
                background: ColorRgba::rgb(0, 0, 0),
                surface: ColorRgba::rgb(10, 10, 10),
                border: ColorRgba::rgb(255, 255, 255),
                text_primary: ColorRgba::rgb(255, 255, 255),
                text_secondary: ColorRgba::rgb(255, 255, 0),
                accent: ColorRgba::rgb(255, 255, 0),
                error: ColorRgba::rgb(255, 50, 50),
                success: ColorRgba::rgb(50, 255, 50),
            },
            ThemeMode::HighContrastLight => Self {
                background: ColorRgba::rgb(255, 255, 255),
                surface: ColorRgba::rgb(255, 255, 255),
                border: ColorRgba::rgb(0, 0, 0),
                text_primary: ColorRgba::rgb(0, 0, 0),
                text_secondary: ColorRgba::rgb(0, 0, 180),
                accent: ColorRgba::rgb(0, 0, 220),
                error: ColorRgba::rgb(180, 0, 0),
                success: ColorRgba::rgb(0, 120, 0),
            },
        }
    }
}

/// Viewport and touch screen constraints for responsive layouts.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ScreenBreakpoints {
    pub is_mobile: bool,
    pub is_mobile_portrait: bool,
    pub is_tablet: bool,
    pub is_desktop: bool,
}

impl ScreenBreakpoints {
    pub fn compute(width: f64, height: f64) -> Self {
        Self {
            is_mobile: width < 768.0,
            is_mobile_portrait: width < 650.0 && height >= width,
            is_tablet: width >= 768.0 && width < 1024.0,
            is_desktop: width >= 1024.0,
        }
    }
}
