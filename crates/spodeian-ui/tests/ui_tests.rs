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

#[test]
fn test_accessible_focus_configuration() {
    let ctx = Context::default();
    let prefs = spodeian_tokens::AccessibilityPreferences {
        high_contrast: true,
        large_touch_targets: true,
        reduced_motion: true,
        font_scale_multiplier: 1.3,
        ..Default::default()
    };

    spodeian_ui::configure_accessible_focus(&ctx, &prefs);

    let style = ctx.style_of(egui::Theme::Dark);
    assert_eq!(style.animation_time, 0.0);
    assert!(style.spacing.interact_size.x >= 44.0);
    assert!(style.spacing.interact_size.y >= 44.0);

    let mut output = ctx.run_ui(Default::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let mut edit_prefs = prefs;
            let changed = spodeian_ui::render_accessibility_panel(ui, &mut edit_prefs);
            assert!(!changed);
        });
    });
    output.textures_delta.clear();
}

#[test]
fn test_reactive_frame_pacer() {
    let ctx = Context::default();
    let mut pacer = spodeian_ui::ReactiveFramePacer::new(
        std::time::Duration::from_millis(50),
        std::time::Duration::from_millis(200),
    );

    // Initial state is active because it was just created
    assert!(!pacer.is_idle());

    pacer.pace_frame(&ctx, false);
    // Explicit notify
    pacer.notify_activity();
    assert!(!pacer.is_idle());

    // Wait past active linger
    std::thread::sleep(std::time::Duration::from_millis(60));
    assert!(pacer.is_idle());

    // Pacing when idle requests delayed repaint
    pacer.pace_frame(&ctx, false);
}

#[test]
fn test_log_console_widget() {
    use spodeian_telemetry::{LogEntry, LogRingBuffer};
    use std::sync::Arc;

    let buffer = Arc::new(LogRingBuffer::new(50));
    buffer.push(LogEntry {
        timestamp_millis: 1000,
        level: "INFO".to_string(),
        target: "spodeian::core".to_string(),
        message: "Initialization complete".to_string(),
        fields: vec![("subsystem".to_string(), "audio".to_string())],
    });
    buffer.push(LogEntry {
        timestamp_millis: 1050,
        level: "ERROR".to_string(),
        target: "spodeian::dsp".to_string(),
        message: "Buffer underflow detected".to_string(),
        fields: vec![],
    });

    let mut widget = spodeian_ui::LogConsoleWidget::new(buffer.clone());

    let ctx = Context::default();
    let mut output = ctx.run_ui(Default::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            widget.ui(ui);
        });
    });
    output.textures_delta.clear();

    assert_eq!(buffer.len(), 2);
}

#[test]
fn test_spodeian_widgets_rendering() {
    let ctx = Context::default();
    let mut theme_mode = ThemeMode::Light;
    let mut slider_val = 42.0;

    let card = spodeian_ui::PerformanceMetricCard::new("Sharpe Ratio", "2.45")
        .unit("annualized")
        .delta(0.15, "+15 bps")
        .tooltip("1-year trailing risk-adjusted return");

    let mut output = ctx.run_ui(Default::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let picker = spodeian_ui::ThemePickerWidget::new(&mut theme_mode);
            ui.add(picker);

            card.show(ui);

            let slider = spodeian_ui::AccessibleSlider::new(&mut slider_val, 0.0..=100.0, "Volume")
                .step(1.0);
            ui.add(slider);
        });
    });
    output.textures_delta.clear();

    assert_eq!(card.title, "Sharpe Ratio");
    assert_eq!(card.value, "2.45");
}
