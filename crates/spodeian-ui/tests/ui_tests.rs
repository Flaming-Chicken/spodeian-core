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

#[test]
fn undo_stack_bounded_history_and_redo_branch_clearing() {
    let mut stack = spodeian_ui::UndoStack::new(2);
    assert!(!stack.can_undo());
    assert!(!stack.can_redo());

    let mut current = 0_i32;
    for next in 1..=3 {
        stack.push(current);
        current = next;
    }
    assert_eq!(current, 3);
    assert_eq!(stack.undo_len(), 2);

    current = stack.undo(current).expect("first undo");
    assert_eq!(current, 2);
    current = stack.undo(current).expect("second undo");
    assert_eq!(current, 1);
    assert!(
        stack.undo(current).is_none(),
        "oldest state (0) was evicted"
    );

    current = stack.redo(current).expect("first redo");
    assert_eq!(current, 2);
    assert!(stack.can_redo());

    stack.push(current);
    current = 99;
    assert_eq!(current, 99);
    assert!(!stack.can_redo(), "push must clear redo branch");

    stack.clear();
    assert!(!stack.can_undo());
    assert!(!stack.can_redo());
}

#[test]
fn command_palette_filters_and_executes_via_keyboard() {
    let commands = vec![
        spodeian_ui::CommandEntry::new("file.save", "Save Project", "Ctrl+S", "File"),
        spodeian_ui::CommandEntry::new("view.theme", "Toggle Theme", "Ctrl+T", "View"),
        spodeian_ui::CommandEntry::new("help.telemetry", "Open Telemetry", "", "Help"),
    ];
    let mut palette = spodeian_ui::CommandPaletteWidget::new(commands);
    assert!(!palette.is_open());
    assert_eq!(palette.filtered_indices(), vec![0, 1, 2]);

    palette.open();
    assert!(palette.is_open());
    palette.set_query("Ctrl");
    assert_eq!(palette.filtered_indices(), vec![0, 1]);

    let ctx = egui::Context::default();
    let mut nav_input = egui::RawInput::default();
    nav_input.events.push(egui::Event::Key {
        key: egui::Key::ArrowDown,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    });
    nav_input.events.push(egui::Event::Key {
        key: egui::Key::Enter,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    });

    let mut executed = None;
    let mut output = ctx.run_ui(nav_input, |ui| {
        executed = palette.show(ui.ctx());
    });
    output.textures_delta.clear();
    assert_eq!(executed.as_deref(), Some("view.theme"));
    assert!(!palette.is_open());

    palette.open();
    let mut esc_input = egui::RawInput::default();
    esc_input.events.push(egui::Event::Key {
        key: egui::Key::Escape,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    });
    let mut esc_output = ctx.run_ui(esc_input, |ui| {
        assert!(palette.show(ui.ctx()).is_none());
    });
    esc_output.textures_delta.clear();
    assert!(!palette.is_open());
}

#[test]
fn telemetry_dashboard_modal_and_kpi_card_render_cleanly() {
    let data = spodeian_telemetry::TelemetryDashboardData {
        title: "Wave 1 Diagnostics".to_string(),
        timestamp_utc: 1_700_000_000,
        uptime_secs: 42,
        total_operations: 1200,
        success_rate: 99.75,
        throughput_ops_sec: 480.0,
        memory_mb: 34.5,
        active_backend: "IndexedDB".to_string(),
        channels: vec![spodeian_telemetry::TelemetryChannelMetric {
            name: "Persist".to_string(),
            count: 64,
            latency_ms: 0.42,
            status: "Healthy".to_string(),
        }],
    };

    let ctx = egui::Context::default();
    let mut open = true;
    let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
        spodeian_ui::render_telemetry_dashboard_modal(ui.ctx(), &mut open, &data);
        egui::CentralPanel::default().show(ui, |ui| {
            spodeian_ui::render_kpi_card(ui, "Latency", "0.42 ms", "p95");
        });
    });
    output.textures_delta.clear();
    assert!(open);
}
