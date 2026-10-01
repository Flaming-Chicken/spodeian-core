//! Accessibility configuration and preference panels for egui applications.
//!
//! **Stability: Stable** — semver-guaranteed since v1.0

use egui::{Color32, Context, Stroke, Ui};
use spodeian_tokens::{AccessibilityPreferences, ColorblindMode, SpectrogramColormap};

/// Configures high-visibility focus borders and accessibility settings on an egui context.
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
pub fn configure_accessible_focus(ctx: &Context, prefs: &AccessibilityPreferences) {
    for theme in [egui::Theme::Dark, egui::Theme::Light] {
        ctx.style_mut_of(theme, |style| {
            if prefs.high_contrast {
                style.visuals.widgets.hovered.bg_stroke = Stroke::new(2.5, Color32::from_rgb(255, 230, 0));
                style.visuals.widgets.active.bg_stroke = Stroke::new(3.0, Color32::from_rgb(255, 230, 0));
            }

            if prefs.large_touch_targets {
                style.spacing.interact_size.x = style.spacing.interact_size.x.max(44.0);
                style.spacing.interact_size.y = style.spacing.interact_size.y.max(44.0);
                style.spacing.button_padding = egui::vec2(12.0, 10.0);
            }

            if prefs.reduced_motion {
                style.animation_time = 0.0;
            }
        });
    }

    if (prefs.font_scale_multiplier - 1.0).abs() > 0.01 {
        let zoom = prefs.font_scale_multiplier.clamp(0.8, 2.5);
        ctx.set_pixels_per_point(zoom);
    }
}

/// Renders an interactive settings panel for adjusting accessibility preferences.
///
/// Returns `true` if any preference value was modified during the frame.
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
pub fn render_accessibility_panel(ui: &mut Ui, prefs: &mut AccessibilityPreferences) -> bool {
    let mut changed = false;

    ui.group(|ui| {
        ui.heading("Accessibility Preferences");
        ui.add_space(4.0);

        if ui
            .checkbox(&mut prefs.high_contrast, "High Contrast Mode")
            .on_hover_text("Enhances boundary contrast and maximizes readability")
            .changed()
        {
            changed = true;
        }

        ui.horizontal(|ui| {
            ui.label("Colorblind Palette:");
            egui::ComboBox::from_id_salt("colorblind_mode_selector")
                .selected_text(format!("{:?}", prefs.colorblind_mode))
                .show_ui(ui, |ui| {
                    for mode in [
                        ColorblindMode::None,
                        ColorblindMode::Deuteranopia,
                        ColorblindMode::Protanopia,
                        ColorblindMode::Tritanopia,
                        ColorblindMode::Monochrome,
                    ] {
                        let is_selected = prefs.colorblind_mode == mode;
                        if ui.selectable_label(is_selected, format!("{mode:?}")).clicked() {
                            prefs.colorblind_mode = mode;
                            changed = true;
                        }
                    }
                });
        });

        ui.horizontal(|ui| {
            ui.label("Audio Spectrogram Colormap:");
            egui::ComboBox::from_id_salt("spectrogram_colormap_selector")
                .selected_text(format!("{:?}", prefs.spectrogram_colormap))
                .show_ui(ui, |ui| {
                    for map in [
                        SpectrogramColormap::RainStandard,
                        SpectrogramColormap::Viridis,
                        SpectrogramColormap::Cividis,
                        SpectrogramColormap::HighContrastMonochrome,
                    ] {
                        let is_selected = prefs.spectrogram_colormap == map;
                        if ui.selectable_label(is_selected, format!("{map:?}")).clicked() {
                            prefs.spectrogram_colormap = map;
                            changed = true;
                        }
                    }
                });
        });

        if ui
            .checkbox(&mut prefs.reduced_motion, "Reduce Motion & Animations")
            .on_hover_text("Disables fluid transitions and animated smoothing")
            .changed()
        {
            changed = true;
        }

        if ui
            .checkbox(&mut prefs.large_touch_targets, "Large Interactive Targets")
            .on_hover_text("Enlarges buttons and controls to at least 44x44 points for touch/motor accessibility")
            .changed()
        {
            changed = true;
        }

        ui.horizontal(|ui| {
            ui.label("UI Scale Multiplier:");
            if ui
                .add(egui::Slider::new(&mut prefs.font_scale_multiplier, 0.8..=2.0).step_by(0.1))
                .changed()
            {
                changed = true;
            }
        });
    });

    changed
}
