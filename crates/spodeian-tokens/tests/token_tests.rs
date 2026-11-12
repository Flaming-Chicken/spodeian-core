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

const ALL_MODES: [spodeian_tokens::ThemeMode; 4] = [
    spodeian_tokens::ThemeMode::Dark,
    spodeian_tokens::ThemeMode::Light,
    spodeian_tokens::ThemeMode::HighContrastDark,
    spodeian_tokens::ThemeMode::HighContrastLight,
];

#[test]
fn test_every_palette_text_and_status_colours_meet_wcag_aa() {
    use spodeian_tokens::{ColorRgba, ThemePalette};

    for mode in ALL_MODES {
        let p = ThemePalette::for_mode(mode);
        for (role, fg) in [
            ("text_primary", p.text_primary),
            ("text_secondary", p.text_secondary),
            ("accent", p.accent),
            ("error", p.error),
            ("success", p.success),
        ] {
            for (surface, bg) in [("background", p.background), ("surface", p.surface)] {
                let ratio = ColorRgba::contrast_ratio(fg, bg);
                assert!(
                    ratio >= 4.5,
                    "{mode:?}: {role} on {surface} is {ratio:.2}:1, below WCAG AA 4.5:1"
                );
            }
        }
    }
}

#[test]
fn test_high_contrast_palettes_meet_wcag_aaa_for_text() {
    use spodeian_tokens::{ColorRgba, ThemeMode, ThemePalette};

    for mode in [ThemeMode::HighContrastDark, ThemeMode::HighContrastLight] {
        let p = ThemePalette::for_mode(mode);
        assert!(ColorRgba::is_wcag_aaa(p.text_primary, p.background));
        assert!(ColorRgba::is_wcag_aaa(p.text_secondary, p.background));
    }
}

#[test]
fn test_color_to_css() {
    use spodeian_tokens::ColorRgba;

    assert_eq!(ColorRgba::rgb(0, 100, 225).to_css(), "rgb(0, 100, 225)");
    assert_eq!(ColorRgba::rgba(1, 2, 3, 0).to_css(), "rgba(1, 2, 3, 0)");
    assert_eq!(ColorRgba::rgba(1, 2, 3, 51).to_css(), "rgba(1, 2, 3, 0.2)");
}

#[test]
fn test_palette_css_custom_properties_cover_every_role() {
    use spodeian_tokens::{ThemeMode, ThemePalette};

    let css = ThemePalette::for_mode(ThemeMode::Dark).css_custom_properties("c-");
    for name in [
        "background",
        "surface",
        "border",
        "text-primary",
        "text-secondary",
        "accent",
        "error",
        "success",
    ] {
        assert!(css.contains(&format!("--c-{name}: rgb(")), "missing {name}");
    }
    assert_eq!(css.lines().count(), 8);
    assert!(css.contains("--c-background: rgb(18, 18, 18);"));
}

#[test]
fn test_theme_stylesheet_has_a_block_per_mode() {
    use spodeian_tokens::{ThemePalette, theme_stylesheet};

    let css = theme_stylesheet("");
    assert!(css.starts_with(":root, [data-theme=\"dark\"] {"));
    for mode in ALL_MODES {
        let selector = format!("[data-theme=\"{}\"]", mode.as_str());
        assert_eq!(css.matches(&selector).count(), 1, "{selector}");
        let text = ThemePalette::for_mode(mode).text_primary.to_css();
        assert!(css.contains(&format!("--text-primary: {text};")));
    }
    assert_eq!(css.matches("color-scheme: dark;").count(), 2);
    assert_eq!(css.matches("color-scheme: light;").count(), 2);
    assert_eq!(css.matches('{').count(), css.matches('}').count());
}

#[test]
fn test_motion_and_font_scale_css() {
    use spodeian_tokens::{AccessibilityPreferences, FontScale, MotionPreference};

    assert_eq!(MotionPreference::default(), MotionPreference::Full);
    assert_eq!(
        MotionPreference::Full.css_custom_properties(""),
        "--motion-scale: 1;\n"
    );
    assert!(MotionPreference::system_preference_css("").contains("prefers-reduced-motion: reduce"));

    assert_eq!(FontScale::new(0.1).value(), FontScale::MIN);
    assert_eq!(FontScale::default().value(), 1.0);

    let prefs = AccessibilityPreferences {
        reduced_motion: true,
        font_scale_multiplier: 1.5,
        ..Default::default()
    };
    assert_eq!(prefs.motion(), MotionPreference::Reduced);
    assert_eq!(
        prefs.css_custom_properties("a-"),
        "--a-motion-scale: 0;\n--a-font-scale: 1.5;\n"
    );
}
