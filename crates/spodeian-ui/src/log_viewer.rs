//! Diagnostics and log viewer widget for egui desktop and web shells.
//!
//! **Stability: Stable** — semver-guaranteed since v1.0

use egui::{Color32, RichText, ScrollArea, TextEdit, Ui};
use spodeian_telemetry::{LogEntry, LogRingBuffer};
use std::sync::Arc;

/// Diagnostic log level filter for the UI console.
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LogLevelFilter {
    /// Show all captured entries
    #[default]
    All,
    /// Filter to TRACE and above
    Trace,
    /// Filter to DEBUG and above
    Debug,
    /// Filter to INFO and above
    Info,
    /// Filter to WARN and above
    Warn,
    /// Filter to ERROR only
    Error,
}

impl LogLevelFilter {
    fn matches(&self, level_str: &str) -> bool {
        let entry_level = match level_str {
            "ERROR" => 5,
            "WARN" => 4,
            "INFO" => 3,
            "DEBUG" => 2,
            "TRACE" => 1,
            _ => 0,
        };

        let min_level = match self {
            LogLevelFilter::All => 0,
            LogLevelFilter::Trace => 1,
            LogLevelFilter::Debug => 2,
            LogLevelFilter::Info => 3,
            LogLevelFilter::Warn => 4,
            LogLevelFilter::Error => 5,
        };

        entry_level >= min_level
    }
}

/// Interactive egui widget for inspecting telemetry logs in real time.
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
#[derive(Debug)]
pub struct LogConsoleWidget {
    buffer: Arc<LogRingBuffer>,
    filter_level: LogLevelFilter,
    search_query: String,
    auto_scroll: bool,
}

impl LogConsoleWidget {
    /// Creates a new log console widget bound to the specified ring buffer.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
    pub fn new(buffer: Arc<LogRingBuffer>) -> Self {
        Self {
            buffer,
            filter_level: LogLevelFilter::All,
            search_query: String::new(),
            auto_scroll: true,
        }
    }

    /// Renders the log console interface.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
    pub fn ui(&mut self, ui: &mut Ui) {
        ui.vertical(|ui| {
            // Control toolbar
            ui.horizontal(|ui| {
                ui.label("Filter:");
                for (name, filter) in [
                    ("All", LogLevelFilter::All),
                    ("Info+", LogLevelFilter::Info),
                    ("Warn+", LogLevelFilter::Warn),
                    ("Error", LogLevelFilter::Error),
                ] {
                    if ui.selectable_label(self.filter_level == filter, name).clicked() {
                        self.filter_level = filter;
                    }
                }

                ui.separator();
                ui.label("Search:");
                ui.add(TextEdit::singleline(&mut self.search_query).hint_text("Filter message..."));

                ui.separator();
                ui.checkbox(&mut self.auto_scroll, "Auto-scroll");

                if ui.button("Clear").clicked() {
                    self.buffer.clear();
                }
            });

            ui.separator();

            let snapshot = self.buffer.snapshot();
            let query = self.search_query.to_lowercase();

            let filtered_entries: Vec<&LogEntry> = snapshot
                .iter()
                .filter(|entry| {
                    if !self.filter_level.matches(&entry.level) {
                        return false;
                    }
                    if !query.is_empty() {
                        let matches_msg = entry.message.to_lowercase().contains(&query);
                        let matches_target = entry.target.to_lowercase().contains(&query);
                        if !matches_msg && !matches_target {
                            return false;
                        }
                    }
                    true
                })
                .collect();

            let scroll_area = ScrollArea::vertical().stick_to_bottom(self.auto_scroll);

            scroll_area.show(ui, |ui| {
                if filtered_entries.is_empty() {
                    ui.weak("No logs matching criteria.");
                } else {
                    for entry in filtered_entries {
                        render_entry_row(ui, entry);
                    }
                }
            });
        });
    }
}

fn level_color(level: &str) -> Color32 {
    match level {
        "ERROR" => Color32::from_rgb(255, 90, 90),
        "WARN" => Color32::from_rgb(240, 185, 45),
        "INFO" => Color32::from_rgb(90, 180, 255),
        "DEBUG" => Color32::from_rgb(180, 180, 180),
        _ => Color32::from_rgb(130, 130, 130),
    }
}

fn render_entry_row(ui: &mut Ui, entry: &LogEntry) {
    ui.horizontal(|ui| {
        ui.colored_label(level_color(&entry.level), format!("[{:<5}]", entry.level));
        ui.label(RichText::new(&entry.target).weak().monospace());
        ui.label(RichText::new(&entry.message).monospace());

        if !entry.fields.is_empty() {
            let fields_str = entry
                .fields
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(" ");
            ui.label(RichText::new(format!("({fields_str})")).weak().italics());
        }
    });
}
