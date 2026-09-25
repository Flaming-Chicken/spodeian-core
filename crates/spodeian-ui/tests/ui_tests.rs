use egui::Context;
use spodeian_ui::{ScreenConstraints, ThemeMode, apply_theme};

#[test]
fn test_screen_constraints_computation() {
    let desktop = ScreenConstraints::from_dimensions(1200.0, 800.0);
    assert!(!desktop.is_mobile);
    assert!(!desktop.is_mobile_portrait);
    assert!(!desktop.is_tight_height);
    assert!(!desktop.is_ultra_tight);

    let mobile = ScreenConstraints::from_dimensions(375.0, 667.0);
    assert!(mobile.is_mobile);
    assert!(mobile.is_mobile_portrait);

    let ultra_tight = ScreenConstraints::from_dimensions(300.0, 400.0);
    assert!(ultra_tight.is_mobile);
    assert!(ultra_tight.is_tight_height);
    assert!(ultra_tight.is_ultra_tight);
}

#[test]
fn test_theme_mode_application() {
    let ctx = Context::default();

    apply_theme(&ctx, ThemeMode::Light);
    assert!(!ThemeMode::Light.is_dark());

    apply_theme(&ctx, ThemeMode::Dark);
    assert!(ThemeMode::Dark.is_dark());

    apply_theme(&ctx, ThemeMode::HighContrastDark);
    assert!(ThemeMode::HighContrastDark.is_dark());
    assert!(ThemeMode::HighContrastDark.is_high_contrast());

    apply_theme(&ctx, ThemeMode::HighContrastLight);
    assert!(!ThemeMode::HighContrastLight.is_dark());
    assert!(ThemeMode::HighContrastLight.is_high_contrast());
}
