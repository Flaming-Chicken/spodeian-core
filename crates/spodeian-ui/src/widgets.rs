//! Universal accessible UI widgets for Spodeian applications.
//!
//! **Stability: Stable** — semver-guaranteed since v1.0

use egui::{Color32, Response, RichText, Ui, Vec2, Widget};
use spodeian_tokens::ThemeMode;

/// Interactive theme mode selection widget supporting light, dark, and high-contrast modes.
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
#[derive(Debug)]
pub struct ThemePickerWidget<'a> {
    current_mode: &'a mut ThemeMode,
}

impl<'a> ThemePickerWidget<'a> {
    pub fn new(current_mode: &'a mut ThemeMode) -> Self {
        Self { current_mode }
    }
}

impl Widget for ThemePickerWidget<'_> {
    fn ui(self, ui: &mut Ui) -> Response {
        let mut changed = false;
        let response = ui.horizontal(|ui| {
            ui.label(RichText::new("Theme:").strong());

            for (mode, label, icon) in [
                (ThemeMode::Light, "Light", "☀️"),
                (ThemeMode::Dark, "Dark", "🌙"),
                (ThemeMode::HighContrastDark, "HC Dark", "🔲"),
                (ThemeMode::HighContrastLight, "HC Light", "🔳"),
            ] {
                let is_selected = *self.current_mode == mode;
                let btn_text = format!("{icon} {label}");
                let btn = egui::Button::new(btn_text).selected(is_selected);
                if ui.add(btn).clicked() && !is_selected {
                    *self.current_mode = mode;
                    changed = true;
                }
            }
        });

        let mut res = response.response;
        if changed {
            res.mark_changed();
        }
        res
    }
}

/// Accessible metric card displaying performance, financial, or DSP telemetry.
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
#[derive(Debug, Clone)]
pub struct PerformanceMetricCard {
    pub title: String,
    pub value: String,
    pub unit: Option<String>,
    pub delta: Option<(f64, String)>,
    pub tooltip: Option<String>,
    pub is_warning: bool,
}

impl PerformanceMetricCard {
    pub fn new(title: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            value: value.into(),
            unit: None,
            delta: None,
            tooltip: None,
            is_warning: false,
        }
    }

    pub fn unit(mut self, unit: impl Into<String>) -> Self {
        self.unit = Some(unit.into());
        self
    }

    pub fn delta(mut self, change_value: f64, formatted: impl Into<String>) -> Self {
        self.delta = Some((change_value, formatted.into()));
        self
    }

    pub fn tooltip(mut self, tooltip: impl Into<String>) -> Self {
        self.tooltip = Some(tooltip.into());
        self
    }

    pub fn warning(mut self, is_warning: bool) -> Self {
        self.is_warning = is_warning;
        self
    }

    pub fn show(&self, ui: &mut Ui) -> Response {
        let frame = egui::Frame::group(ui.style()).inner_margin(8.0);
        let resp = frame
            .show(ui, |ui| {
                ui.set_min_size(Vec2::new(140.0, 70.0));
                ui.vertical(|ui| {
                    ui.label(RichText::new(&self.title).size(11.0).weak());
                    ui.add_space(2.0);

                    ui.horizontal(|ui| {
                        let val_color = if self.is_warning {
                            Color32::from_rgb(230, 80, 80)
                        } else {
                            ui.visuals().text_color()
                        };
                        ui.label(RichText::new(&self.value).size(18.0).strong().color(val_color));

                        if let Some(unit) = &self.unit {
                            ui.label(RichText::new(unit).size(12.0).weak());
                        }
                    });

                    if let Some((val, text)) = &self.delta {
                        ui.add_space(2.0);
                        let (arrow, color) = if *val > 0.0 {
                            ("▲", Color32::from_rgb(50, 180, 80))
                        } else if *val < 0.0 {
                            ("▼", Color32::from_rgb(220, 70, 70))
                        } else {
                            ("•", Color32::GRAY)
                        };
                        ui.label(RichText::new(format!("{arrow} {text}")).size(10.0).color(color));
                    }
                });
            })
            .response;

        if let Some(tip) = &self.tooltip {
            resp.on_hover_text(tip)
        } else {
            resp
        }
    }
}

/// Accessible high-contrast slider widget with keyboard interaction and touch targets.
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
pub struct AccessibleSlider<'a> {
    value: &'a mut f64,
    range: std::ops::RangeInclusive<f64>,
    label: String,
    step: Option<f64>,
}

impl<'a> AccessibleSlider<'a> {
    pub fn new(value: &'a mut f64, range: std::ops::RangeInclusive<f64>, label: impl Into<String>) -> Self {
        Self {
            value,
            range,
            label: label.into(),
            step: None,
        }
    }

    pub fn step(mut self, step: f64) -> Self {
        self.step = Some(step);
        self
    }
}

impl Widget for AccessibleSlider<'_> {
    fn ui(self, ui: &mut Ui) -> Response {
        ui.horizontal(|ui| {
            ui.label(&self.label);
            let mut slider = egui::Slider::new(self.value, self.range.clone());
            if let Some(step) = self.step {
                slider = slider.step_by(step);
            }
            ui.add(slider)
        })
        .inner
    }
}
