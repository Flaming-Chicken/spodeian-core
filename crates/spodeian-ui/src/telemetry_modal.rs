//! Reusable egui performance telemetry dashboard modal and KPI metric card helper.

use egui::{Color32, Context, RichText, ScrollArea, Ui};
use spodeian_telemetry::TelemetryDashboardData;
use spodeian_web_utils::copy_to_clipboard;

/// Renders the shared performance telemetry dashboard modal window with KPI summary cards,
/// per-channel throughput/latency table, and JSON snapshot clipboard export.
pub fn render_telemetry_dashboard_modal(
    ctx: &Context,
    open: &mut bool,
    data: &TelemetryDashboardData,
) {
    if !*open {
        return;
    }

    let mut is_open = *open;
    let mut close_clicked = false;

    egui::Window::new("System Performance Telemetry")
        .open(&mut is_open)
        .collapsible(false)
        .resizable(true)
        .default_width(580.0)
        .show(ctx, |ui| {
            ui.add_space(4.0);

            ui.horizontal(|ui| {
                ui.heading(&data.title);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let badge_text = if data.success_rate > 99.0 {
                        "● System Operational"
                    } else {
                        "▲ Degraded"
                    };
                    let badge_color = if data.success_rate > 99.0 {
                        Color32::from_rgb(70, 190, 110)
                    } else {
                        Color32::from_rgb(220, 150, 40)
                    };
                    ui.colored_label(badge_color, badge_text);
                });
            });

            if !data.active_backend.is_empty() {
                ui.label(
                    RichText::new(format!("Storage Backend: {}", data.active_backend))
                        .size(11.0)
                        .weak(),
                );
            }

            ui.separator();
            ui.add_space(4.0);

            egui::Grid::new("spodeian_telemetry_kpi_grid")
                .num_columns(4)
                .spacing([12.0, 10.0])
                .show(ui, |ui| {
                    render_kpi_card(
                        ui,
                        "Uptime",
                        &format!("{}s", data.uptime_secs),
                        "Online Duration",
                    );
                    render_kpi_card(
                        ui,
                        "Operations",
                        &format!("{}", data.total_operations),
                        "Total Invocations",
                    );
                    render_kpi_card(
                        ui,
                        "Success Rate",
                        &format!("{:.2}%", data.success_rate),
                        "Zero-Fault SLA",
                    );
                    render_kpi_card(
                        ui,
                        "Throughput",
                        &format!("{:.0} ops/s", data.throughput_ops_sec),
                        "Peak Velocity",
                    );
                    ui.end_row();
                });

            ui.add_space(8.0);
            ui.separator();
            ui.add_space(4.0);

            ui.label(RichText::new("Subsystem Channels & Health Telemetry").strong());
            ui.add_space(2.0);

            ScrollArea::vertical().max_height(200.0).show(ui, |ui| {
                egui::Grid::new("spodeian_telemetry_channels_grid")
                    .striped(true)
                    .num_columns(4)
                    .spacing([24.0, 6.0])
                    .show(ui, |ui| {
                        ui.label(RichText::new("Channel").strong());
                        ui.label(RichText::new("Processed").strong());
                        ui.label(RichText::new("Latency").strong());
                        ui.label(RichText::new("Status").strong());
                        ui.end_row();

                        for ch in &data.channels {
                            ui.label(&ch.name);
                            ui.label(format!("{}", ch.count));
                            ui.label(format!("{:.2} ms", ch.latency_ms));
                            let color = if ch.status == "Healthy" || ch.status == "Optimal" {
                                Color32::from_rgb(70, 190, 110)
                            } else {
                                Color32::from_rgb(220, 150, 40)
                            };
                            ui.colored_label(color, &ch.status);
                            ui.end_row();
                        }
                    });
            });

            ui.add_space(10.0);
            ui.separator();

            ui.horizontal(|ui| {
                if ui.button("Copy Telemetry Snapshot").clicked()
                    && let Ok(json) = serde_json::to_string_pretty(data)
                {
                    copy_to_clipboard(&json);
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Close").clicked() {
                        close_clicked = true;
                    }
                });
            });
        });

    *open = is_open && !close_clicked;
}

/// Renders a single framed KPI summary tile inside telemetry dashboards.
pub fn render_kpi_card(ui: &mut Ui, title: &str, value: &str, subtitle: &str) {
    egui::Frame::group(ui.style())
        .inner_margin(egui::Margin::symmetric(10, 8))
        .show(ui, |ui| {
            ui.vertical(|ui| {
                ui.label(RichText::new(title).size(11.0).weak());
                ui.label(RichText::new(value).size(17.0).strong());
                ui.label(RichText::new(subtitle).size(10.0).weak());
            });
        });
}
