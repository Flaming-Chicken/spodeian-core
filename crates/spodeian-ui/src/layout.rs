//! Responsive layout constraints and breakpoints for egui applications.

use egui::Ui;

/// Dynamic screen and viewport constraints derived from available dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScreenConstraints {
    pub is_mobile: bool,
    pub is_mobile_portrait: bool,
    pub is_tight_height: bool,
    pub is_ultra_tight: bool,
}

impl ScreenConstraints {
    /// Computes constraints from available Ui dimensions.
    pub fn compute(ui: &Ui) -> Self {
        Self::from_dimensions(ui.available_width(), ui.available_height())
    }

    /// Computes constraints from explicit width and height.
    pub fn from_dimensions(avail_w: f32, avail_h: f32) -> Self {
        Self {
            is_mobile: avail_w < 800.0,
            is_mobile_portrait: avail_w < 650.0,
            is_tight_height: avail_h < 530.0 || avail_w < 350.0,
            is_ultra_tight: avail_w < 330.0 || avail_h < 490.0,
        }
    }
}
