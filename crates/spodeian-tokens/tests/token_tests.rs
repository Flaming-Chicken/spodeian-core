use spodeian_tokens::{ColorRgba, ScreenBreakpoints, ThemeMode, ThemePalette};

#[test]
fn test_wcag_contrast_ratios_standard_and_high_contrast() {
    let dark_palette = ThemePalette::for_mode(ThemeMode::Dark);
    assert!(
        ColorRgba::is_wcag_aa(dark_palette.text_primary, dark_palette.background),
        "Dark theme primary text must satisfy WCAG AA (>= 4.5:1)"
    );

    let light_palette = ThemePalette::for_mode(ThemeMode::Light);
    assert!(
        ColorRgba::is_wcag_aa(light_palette.text_primary, light_palette.background),
        "Light theme primary text must satisfy WCAG AA (>= 4.5:1)"
    );

    let hc_dark = ThemePalette::for_mode(ThemeMode::HighContrastDark);
    assert!(
        ColorRgba::is_wcag_aaa(hc_dark.text_primary, hc_dark.background),
        "High-contrast dark must satisfy WCAG AAA (>= 7.0:1)"
    );

    let hc_light = ThemePalette::for_mode(ThemeMode::HighContrastLight);
    assert!(
        ColorRgba::is_wcag_aaa(hc_light.text_primary, hc_light.background),
        "High-contrast light must satisfy WCAG AAA (>= 7.0:1)"
    );
}

#[test]
fn test_screen_breakpoint_calculations() {
    // Mobile portrait
    let bp_phone = ScreenBreakpoints::compute(390.0, 844.0);
    assert!(bp_phone.is_mobile);
    assert!(bp_phone.is_mobile_portrait);
    assert!(!bp_phone.is_desktop);

    // Desktop widescreen
    let bp_desktop = ScreenBreakpoints::compute(1920.0, 1080.0);
    assert!(!bp_desktop.is_mobile);
    assert!(bp_desktop.is_desktop);
    assert!(!bp_desktop.is_tablet);

    // Tablet
    let bp_tablet = ScreenBreakpoints::compute(820.0, 1180.0);
    assert!(bp_tablet.is_tablet);
    assert!(!bp_tablet.is_mobile);
    assert!(!bp_tablet.is_desktop);
}
