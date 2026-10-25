//! Ecosystem Design Tokens, WCAG 2.1 Accessibility & Typography (`spodeian-tokens`).
//!
//! **Stability: Stable** — semver-guaranteed since v1.0
//!
//! Provides unified themes, color palettes, responsive screen breakpoints, and KaTeX rendering options
//! shared across Leptos (web) and egui (desktop/mobile) projects.

use serde::{Deserialize, Serialize};

#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ThemeMode {
    #[default]
    Dark,
    Light,
    HighContrastDark,
    HighContrastLight,
}

impl ThemeMode {
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
    pub fn is_high_contrast(self) -> bool {
        matches!(self, Self::HighContrastDark | Self::HighContrastLight)
    }

    #[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
    pub fn is_dark(self) -> bool {
        matches!(self, Self::Dark | Self::HighContrastDark)
    }

    #[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
    pub fn next(self) -> Self {
        match self {
            Self::Dark => Self::Light,
            Self::Light => Self::HighContrastDark,
            Self::HighContrastDark => Self::HighContrastLight,
            Self::HighContrastLight => Self::Dark,
        }
    }

    #[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
    pub fn toggle(self) -> Self {
        self.next()
    }

    #[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Dark => "dark",
            Self::Light => "light",
            Self::HighContrastDark => "high-contrast-dark",
            Self::HighContrastLight => "high-contrast-light",
        }
    }

    #[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        match s {
            "light" => Self::Light,
            "high-contrast-dark" => Self::HighContrastDark,
            "high-contrast-light" => Self::HighContrastLight,
            _ => Self::Dark,
        }
    }

    #[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
    pub fn display_label(self) -> &'static str {
        match self {
            Self::Dark => "Dark",
            Self::Light => "Warm Light",
            Self::HighContrastDark => "HC Dark",
            Self::HighContrastLight => "HC Light",
        }
    }

    #[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
    pub fn label(self) -> &'static str {
        match self {
            Self::Dark => "Dark",
            Self::Light => "Warm Light",
            Self::HighContrastDark => "High Contrast (Dark)",
            Self::HighContrastLight => "High Contrast (Light)",
        }
    }

    #[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
    pub fn icon(self) -> &'static str {
        match self {
            Self::Dark => "🌙",
            Self::Light => "☀️",
            Self::HighContrastDark => "⬛",
            Self::HighContrastLight => "⬜",
        }
    }
}

impl std::str::FromStr for ThemeMode {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self::from_str(s))
    }
}

/// Standard 8-bit RGBA color representation with WCAG luminance calculations.
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ColorRgba {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl ColorRgba {
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }

    #[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
    pub const fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Computes the relative luminance according to WCAG 2.1 specifications:
    /// $L = 0.2126 R + 0.7152 G + 0.0722 B$
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
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
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
    pub fn contrast_ratio(c1: Self, c2: Self) -> f64 {
        let l1 = c1.relative_luminance();
        let l2 = c2.relative_luminance();
        let lighter = l1.max(l2);
        let darker = l1.min(l2);
        (lighter + 0.05) / (darker + 0.05)
    }

    /// Checks if contrast ratio meets WCAG 2.1 AA level for standard text ($\ge 4.5:1$).
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
    pub fn is_wcag_aa(c1: Self, c2: Self) -> bool {
        Self::contrast_ratio(c1, c2) >= 4.5
    }

    /// Checks if contrast ratio meets WCAG 2.1 AAA level for high contrast ($\ge 7.0:1$).
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
    pub fn is_wcag_aaa(c1: Self, c2: Self) -> bool {
        Self::contrast_ratio(c1, c2) >= 7.0
    }
}

/// Palette tokens for a UI theme.
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
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
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
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
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ScreenBreakpoints {
    pub is_mobile: bool,
    pub is_mobile_portrait: bool,
    pub is_tablet: bool,
    pub is_desktop: bool,
}

impl ScreenBreakpoints {
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
    pub fn compute(width: f64, height: f64) -> Self {
        Self {
            is_mobile: width < 768.0,
            is_mobile_portrait: width < 650.0 && height >= width,
            is_tablet: (768.0..1024.0).contains(&width),
            is_desktop: width >= 1024.0,
        }
    }
}

/// Colorblind vision profile modes for financial charts and visual indicators.
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ColorblindMode {
    /// Default green-up / red-down standard trichromatic palette.
    #[default]
    None,
    /// Blue-up / Orange-down optimized for Deuteranopia (green-weak).
    Deuteranopia,
    /// Blue-up / Vermillion-down calibrated for Protanopia (red-weak).
    Protanopia,
    /// Cyan-up / Magenta-down calibrated for Tritanopia (blue-yellow).
    Tritanopia,
    /// High-contrast monochromatic scale for Achromatopsia or direct printing.
    Monochrome,
}

/// Colorblind-aware candlestick palette for financial market visualizations.
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandlestickColors {
    pub up: ColorRgba,
    pub down: ColorRgba,
    pub neutral: ColorRgba,
}

impl CandlestickColors {
    /// Returns candlestick colors calibrated for the specified vision profile.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
    pub fn for_mode(mode: ColorblindMode) -> Self {
        match mode {
            ColorblindMode::None => Self {
                up: ColorRgba::rgb(34, 197, 94),   // Emerald green
                down: ColorRgba::rgb(239, 68, 68), // Crimson red
                neutral: ColorRgba::rgb(156, 163, 175),
            },
            ColorblindMode::Deuteranopia => Self {
                up: ColorRgba::rgb(59, 130, 246),   // Bright blue
                down: ColorRgba::rgb(249, 115, 22), // High-visibility orange
                neutral: ColorRgba::rgb(156, 163, 175),
            },
            ColorblindMode::Protanopia => Self {
                up: ColorRgba::rgb(37, 99, 235),   // Blue
                down: ColorRgba::rgb(234, 88, 12), // Vermillion
                neutral: ColorRgba::rgb(156, 163, 175),
            },
            ColorblindMode::Tritanopia => Self {
                up: ColorRgba::rgb(14, 165, 233),  // Cyan
                down: ColorRgba::rgb(244, 63, 94), // Rose / Magenta
                neutral: ColorRgba::rgb(156, 163, 175),
            },
            ColorblindMode::Monochrome => Self {
                up: ColorRgba::rgb(243, 244, 246), // Off-white
                down: ColorRgba::rgb(55, 65, 81),  // Dark slate
                neutral: ColorRgba::rgb(156, 163, 175),
            },
        }
    }
}

/// Perceptually uniform and colorblind-safe spectrogram colormaps.
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SpectrogramColormap {
    /// Procedural rain cyan/blue/indigo scale.
    #[default]
    RainStandard,
    /// Perceptually uniform Viridis colormap (yellow-green-purple).
    Viridis,
    /// Cividis colormap (scientifically optimized for red-green colorblindness).
    Cividis,
    /// High-contrast monochrome gradient (pure black to pure white).
    HighContrastMonochrome,
}

impl SpectrogramColormap {
    /// Evaluates the colormap at normalized intensity $t \in [0.0, 1.0]$.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
    pub fn sample(self, t: f64) -> ColorRgba {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::RainStandard => {
                let r = (20.0 * (1.0 - t) + 120.0 * t) as u8;
                let g = (40.0 * (1.0 - t) + 220.0 * t) as u8;
                let b = (80.0 * (1.0 - t) + 255.0 * t) as u8;
                ColorRgba::rgb(r, g, b)
            }
            Self::Viridis => {
                // Piecewise linear approximation of Viridis
                let r = (68.0 + 185.0 * t.powf(1.5)).min(253.0) as u8;
                let g = (1.0 + 230.0 * t.sin()).min(231.0) as u8;
                let b = (84.0 + 100.0 * (1.0 - t).powi(2) + 37.0 * t).min(255.0) as u8;
                ColorRgba::rgb(r, g, b)
            }
            Self::Cividis => {
                // Perceptually uniform blue-gray-yellow for color vision deficiency
                let r = (0.0 + 250.0 * t) as u8;
                let g = (32.0 + 200.0 * t) as u8;
                let b = (77.0 + 80.0 * (1.0 - t) + 70.0 * t) as u8;
                ColorRgba::rgb(r, g, b)
            }
            Self::HighContrastMonochrome => {
                let v = (255.0 * t) as u8;
                ColorRgba::rgb(v, v, v)
            }
        }
    }
}

/// Global user accessibility preferences across all Spodeian applications.
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AccessibilityPreferences {
    /// Whether high-contrast mode is forced.
    pub high_contrast: bool,
    /// When true, disables intense animations, particle swirls, and pulsating graphics.
    pub reduced_motion: bool,
    /// Active vision impairment color compensation profile.
    pub colorblind_mode: ColorblindMode,
    /// Active audio spectrogram colormap.
    pub spectrogram_colormap: SpectrogramColormap,
    /// Enlarge touch targets to min 44x44pt.
    pub large_touch_targets: bool,
    /// Font / UI scale multiplier (default 1.0).
    pub font_scale_multiplier: f32,
}

impl Default for AccessibilityPreferences {
    fn default() -> Self {
        Self {
            high_contrast: false,
            reduced_motion: false,
            colorblind_mode: ColorblindMode::None,
            spectrogram_colormap: SpectrogramColormap::RainStandard,
            large_touch_targets: false,
            font_scale_multiplier: 1.0,
        }
    }
}

/// Command-line accessibility and output formatting helper.
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CliAccessibility {
    /// Suppress ANSI colors and unicode borders for screen readers and logging.
    pub plain: bool,
    /// Output machine-readable JSON format.
    pub json: bool,
    /// Disable color escapes (e.g. `NO_COLOR` environment variable or `--no-color` flag).
    pub no_color: bool,
}

impl CliAccessibility {
    /// Detects accessibility settings from CLI flags and system environment variables (`NO_COLOR`).
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
    pub fn new(plain: bool, json: bool, no_color_flag: bool) -> Self {
        let env_no_color = std::env::var_os("NO_COLOR").is_some();
        Self {
            plain,
            json,
            no_color: plain || json || no_color_flag || env_no_color,
        }
    }

    /// Returns true if output should include ANSI color codes.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
    pub fn use_color(&self) -> bool {
        !self.no_color
    }
}
