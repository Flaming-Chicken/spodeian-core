//! Egui visual styles and theme management adhering to WCAG 2.1 AAA contrast.
//!
//! **Stability: Stable** — semver-guaranteed since v1.0

use egui::{Color32, Context, Stroke, Visuals};
pub use spodeian_tokens::ThemeMode;

/// Applies the selected ThemeMode to an egui Context.
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
pub fn apply_theme(ctx: &Context, mode: ThemeMode) {
    let visuals = match mode {
        ThemeMode::Light => {
            let mut light = Visuals::light();

            // Soothing neutral/warm light backgrounds
            light.panel_fill = Color32::from_rgb(245, 244, 241);
            light.window_fill = Color32::from_rgb(252, 250, 246);
            light.extreme_bg_color = Color32::from_rgb(238, 236, 231);

            // Soft charcoal for high-contrast, comfortable reading
            light.widgets.noninteractive.fg_stroke.color = Color32::from_rgb(45, 44, 42);
            light.widgets.inactive.fg_stroke.color = Color32::from_rgb(55, 54, 52);
            light.widgets.hovered.fg_stroke.color = Color32::from_rgb(20, 20, 18);
            light.widgets.active.fg_stroke.color = Color32::from_rgb(0, 0, 0);

            // Muted border strokes
            light.widgets.noninteractive.bg_stroke.color = Color32::from_rgb(222, 220, 215);
            light.widgets.inactive.bg_stroke.color = Color32::from_rgb(212, 210, 205);

            // Buttons background
            light.widgets.inactive.bg_fill = Color32::from_rgb(252, 251, 248);
            light.widgets.hovered.bg_fill = Color32::from_rgb(236, 234, 229);
            light.widgets.active.bg_fill = Color32::from_rgb(220, 218, 212);

            light
        }
        ThemeMode::Dark => Visuals::dark(),
        ThemeMode::HighContrastDark => {
            let mut hc = Visuals::dark();
            hc.panel_fill = Color32::BLACK;
            hc.window_fill = Color32::BLACK;
            hc.extreme_bg_color = Color32::from_rgb(10, 10, 10);
            hc.widgets.noninteractive.fg_stroke = Stroke::new(1.5, Color32::WHITE);
            hc.widgets.inactive.fg_stroke = Stroke::new(1.5, Color32::WHITE);
            hc.widgets.hovered.fg_stroke = Stroke::new(2.0, Color32::from_rgb(255, 255, 0));
            hc.widgets.active.fg_stroke = Stroke::new(2.0, Color32::from_rgb(255, 255, 0));
            hc.widgets.noninteractive.bg_stroke = Stroke::new(2.0, Color32::WHITE);
            hc.widgets.inactive.bg_stroke = Stroke::new(2.0, Color32::WHITE);
            hc.widgets.hovered.bg_stroke = Stroke::new(2.5, Color32::from_rgb(255, 255, 0));
            hc.widgets.active.bg_stroke = Stroke::new(2.5, Color32::from_rgb(255, 255, 0));
            hc.widgets.inactive.bg_fill = Color32::BLACK;
            hc.widgets.hovered.bg_fill = Color32::from_rgb(30, 30, 0);
            hc.widgets.active.bg_fill = Color32::from_rgb(50, 50, 0);
            hc
        }
        ThemeMode::HighContrastLight => {
            let mut hc = Visuals::light();
            hc.panel_fill = Color32::WHITE;
            hc.window_fill = Color32::WHITE;
            hc.extreme_bg_color = Color32::WHITE;
            hc.widgets.noninteractive.fg_stroke = Stroke::new(1.5, Color32::BLACK);
            hc.widgets.inactive.fg_stroke = Stroke::new(1.5, Color32::BLACK);
            hc.widgets.hovered.fg_stroke = Stroke::new(2.0, Color32::from_rgb(0, 0, 180));
            hc.widgets.active.fg_stroke = Stroke::new(2.0, Color32::from_rgb(0, 0, 220));
            hc.widgets.noninteractive.bg_stroke = Stroke::new(2.0, Color32::BLACK);
            hc.widgets.inactive.bg_stroke = Stroke::new(2.0, Color32::BLACK);
            hc.widgets.hovered.bg_stroke = Stroke::new(2.5, Color32::from_rgb(0, 0, 180));
            hc.widgets.active.bg_stroke = Stroke::new(2.5, Color32::from_rgb(0, 0, 220));
            hc.widgets.inactive.bg_fill = Color32::WHITE;
            hc.widgets.hovered.bg_fill = Color32::from_rgb(230, 235, 255);
            hc.widgets.active.bg_fill = Color32::from_rgb(210, 220, 255);
            hc
        }
    };
    ctx.set_visuals(visuals);
}
