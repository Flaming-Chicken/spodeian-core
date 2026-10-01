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

#[test]
fn test_colorblind_candlestick_palettes() {
    use spodeian_tokens::{CandlestickColors, ColorblindMode};

    let standard = CandlestickColors::for_mode(ColorblindMode::None);
    assert_eq!(standard.up, ColorRgba::rgb(34, 197, 94));
    assert_eq!(standard.down, ColorRgba::rgb(239, 68, 68));

    let deut = CandlestickColors::for_mode(ColorblindMode::Deuteranopia);
    assert_eq!(deut.up, ColorRgba::rgb(59, 130, 246));
    assert_eq!(deut.down, ColorRgba::rgb(249, 115, 22));

    let mono = CandlestickColors::for_mode(ColorblindMode::Monochrome);
    assert_eq!(mono.up, ColorRgba::rgb(243, 244, 246));
    assert_eq!(mono.down, ColorRgba::rgb(55, 65, 81));
}

#[test]
fn test_spectrogram_colormaps() {
    use spodeian_tokens::SpectrogramColormap;

    for cmap in [
        SpectrogramColormap::RainStandard,
        SpectrogramColormap::Viridis,
        SpectrogramColormap::Cividis,
        SpectrogramColormap::HighContrastMonochrome,
    ] {
        let low = cmap.sample(0.0);
        let mid = cmap.sample(0.5);
        let high = cmap.sample(1.0);

        // Verify color values are valid
        assert_eq!(low.a, 255);
        assert_eq!(mid.a, 255);
        assert_eq!(high.a, 255);

        // Low and high must differ
        assert_ne!(low, high);
    }
}

#[test]
fn test_cli_accessibility_helpers() {
    use spodeian_tokens::CliAccessibility;

    let cli_plain = CliAccessibility::new(true, false, false);
    assert!(cli_plain.plain);
    assert!(cli_plain.no_color);
    assert!(!cli_plain.use_color());

    let cli_json = CliAccessibility::new(false, true, false);
    assert!(cli_json.json);
    assert!(cli_json.no_color);

    let cli_standard = CliAccessibility::new(false, false, false);
    assert!(!cli_standard.plain);
    assert!(!cli_standard.json);
}

