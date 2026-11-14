//! Shared Web & Leptos UI primitives, accessibility helpers, and view models.
//!
//! Designed to be consumed directly by Leptos frontends (`efficient-leptos-template`,
//! `page-portal`, and future web apps) without imposing a pinned `leptos` patch version
//! on non-web workspace members.

use serde::{Deserialize, Serialize};
use spodeian_telemetry::{TelemetryMetrics, metrics_snapshot};
use spodeian_tokens::ThemeMode;

/// Default `localStorage` key used for persisting the active color theme.
pub const THEME_STORAGE_KEY: &str = "app_theme";

/// Default `localStorage` key used for persisting the dyslexia-friendly typography setting.
pub const DYSLEXIA_STORAGE_KEY: &str = "app_dyslexia";

#[cfg(target_arch = "wasm32")]
const FOCUSABLE_SELECTOR: &str = "a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex=\"-1\"])";

/// Normalizes an arbitrary theme string into one of [`ThemeMode`]'s canonical identifiers
/// (`"dark"`, `"light"`, `"high-contrast-dark"`, or `"high-contrast-light"`).
pub fn normalize_theme(raw: &str) -> &'static str {
    let lower = raw.trim().to_ascii_lowercase();
    ThemeMode::from_str(&lower).as_str()
}

/// Cycles to the next [`ThemeMode`] identifier in display order.
pub fn toggle_theme_value(current: &str) -> &'static str {
    let normalized = normalize_theme(current);
    ThemeMode::from_str(normalized).next().as_str()
}

/// Applies `theme` to `<html data-theme="...">` and saves it under `storage_key` on `wasm32`.
pub fn apply_document_theme_with_key(theme: &str, storage_key: &str) {
    let normalized = normalize_theme(theme);
    #[cfg(target_arch = "wasm32")]
    {
        if let Some(window) = web_sys::window() {
            if let Some(doc) = window.document()
                && let Some(html) = doc.document_element()
            {
                let _ = html.set_attribute("data-theme", normalized);
            }
            if let Ok(Some(storage)) = window.local_storage() {
                let _ = storage.set_item(storage_key, normalized);
            }
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (normalized, storage_key);
    }
}

/// Applies `theme` to the document using [`THEME_STORAGE_KEY`].
pub fn apply_document_theme(theme: &str) {
    apply_document_theme_with_key(theme, THEME_STORAGE_KEY);
}

/// Reads the initial normalized theme identifier from `localStorage[storage_key]` or returns
/// `"dark"`.
pub fn get_initial_theme_with_key(storage_key: &str) -> String {
    #[cfg(target_arch = "wasm32")]
    {
        if let Some(window) = web_sys::window()
            && let Ok(Some(storage)) = window.local_storage()
            && let Ok(Some(val)) = storage.get_item(storage_key)
        {
            return normalize_theme(&val).to_string();
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = storage_key;
    }
    ThemeMode::Dark.as_str().to_string()
}

/// Reads the initial normalized theme identifier using [`THEME_STORAGE_KEY`].
pub fn get_initial_theme() -> String {
    get_initial_theme_with_key(THEME_STORAGE_KEY)
}

/// Applies the dyslexia-friendly typography attribute (`data-dyslexia="true"`) to `<html>`
/// and saves `"true"`/`"false"` in `localStorage[storage_key]` on `wasm32`.
pub fn apply_document_dyslexia_with_key(enabled: bool, storage_key: &str) {
    #[cfg(target_arch = "wasm32")]
    {
        if let Some(window) = web_sys::window() {
            if let Some(doc) = window.document()
                && let Some(html) = doc.document_element()
            {
                if enabled {
                    let _ = html.set_attribute("data-dyslexia", "true");
                } else {
                    let _ = html.remove_attribute("data-dyslexia");
                }
            }
            if let Ok(Some(storage)) = window.local_storage() {
                let _ = storage.set_item(storage_key, if enabled { "true" } else { "false" });
            }
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (enabled, storage_key);
    }
}

/// Applies the dyslexia-friendly typography attribute using [`DYSLEXIA_STORAGE_KEY`].
pub fn apply_document_dyslexia(enabled: bool) {
    apply_document_dyslexia_with_key(enabled, DYSLEXIA_STORAGE_KEY);
}

/// Reads the stored dyslexia typography toggle from `localStorage[storage_key]` or defaults to
/// `false`.
pub fn get_initial_dyslexia_with_key(storage_key: &str) -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        if let Some(window) = web_sys::window()
            && let Ok(Some(storage)) = window.local_storage()
            && let Ok(Some(val)) = storage.get_item(storage_key)
        {
            return val == "true";
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = storage_key;
    }
    false
}

/// Reads the stored dyslexia typography toggle using [`DYSLEXIA_STORAGE_KEY`].
pub fn get_initial_dyslexia() -> bool {
    get_initial_dyslexia_with_key(DYSLEXIA_STORAGE_KEY)
}

/// Target endpoint chosen by [`should_wrap_modal_tab`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TabWrapTarget {
    First,
    Last,
}

/// Determines whether pressing `Tab` (or `Shift+Tab`) at a modal focus boundary should wrap.
pub fn should_wrap_modal_tab(
    shift_key: bool,
    is_first_focused: bool,
    is_last_focused: bool,
) -> Option<TabWrapTarget> {
    if shift_key && is_first_focused {
        Some(TabWrapTarget::Last)
    } else if !shift_key && is_last_focused {
        Some(TabWrapTarget::First)
    } else {
        None
    }
}

/// Traps keyboard `Tab` focus between `first` and `last` focusable endpoints inside a modal
/// dialog. Callers should call `prevent_default()` when this returns `true`.
#[cfg(target_arch = "wasm32")]
pub fn trap_tab_between_endpoints(
    shift_key: bool,
    first: Option<&web_sys::HtmlElement>,
    last: Option<&web_sys::HtmlElement>,
) -> bool {
    let Some(doc) = web_sys::window().and_then(|w| w.document()) else {
        return false;
    };
    let active = doc.active_element();
    let is_first = match (active.as_ref(), first) {
        (Some(active_el), Some(first_el)) => first_el.is_same_node(Some(active_el.as_ref())),
        _ => false,
    };
    let is_last = match (active.as_ref(), last) {
        (Some(active_el), Some(last_el)) => last_el.is_same_node(Some(active_el.as_ref())),
        _ => false,
    };
    match should_wrap_modal_tab(shift_key, is_first, is_last) {
        Some(TabWrapTarget::Last) => {
            if let Some(el) = last {
                let _ = el.focus();
            }
            true
        }
        Some(TabWrapTarget::First) => {
            if let Some(el) = first {
                let _ = el.focus();
            }
            true
        }
        None => false,
    }
}

/// Traps keyboard `Tab` focus within the element identified by `modal_id`.
///
/// Returns `true` when focus wrapped between the first and last focusable descendants (so the
/// caller can `prevent_default()` on the event). Returns `false` outside `wasm32`.
pub fn trap_tab_within_element(modal_id: &str, shift_key: bool) -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        use wasm_bindgen::JsCast;
        let Some(doc) = web_sys::window().and_then(|w| w.document()) else {
            return false;
        };
        let Some(container) = doc.get_element_by_id(modal_id) else {
            return false;
        };
        let Ok(nodes) = container.query_selector_all(FOCUSABLE_SELECTOR) else {
            return false;
        };
        let mut focusable = Vec::new();
        for idx in 0..nodes.length() {
            if let Some(node) = nodes.item(idx)
                && let Ok(html_el) = node.dyn_into::<web_sys::HtmlElement>()
            {
                focusable.push(html_el);
            }
        }
        let first = focusable.first();
        let last = focusable.last();
        trap_tab_between_endpoints(shift_key, first, last)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (modal_id, shift_key);
        false
    }
}

/// Presentation view-model for a dashboard KPI metric card.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KpiCardSpec {
    pub label: String,
    pub value: String,
    pub subtext: String,
    pub accent_color: String,
}

/// Structured shortcut entry displayed in Help and Shortcuts modals.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShortcutEntry {
    pub keys: String,
    pub description: String,
}

/// Standard web application keyboard shortcuts adhering to WCAG 2.1 navigation patterns.
pub fn default_web_shortcuts() -> Vec<ShortcutEntry> {
    vec![
        ShortcutEntry {
            keys: "Enter".to_string(),
            description: "Quickly submit new item creation in form".to_string(),
        },
        ShortcutEntry {
            keys: "Escape".to_string(),
            description: "Dismiss any open modal dialog".to_string(),
        },
        ShortcutEntry {
            keys: "Tab / Shift+Tab".to_string(),
            description: "Cycle focus safely within active dialogs".to_string(),
        },
    ]
}

/// Recoverable error-boundary view model rendered when a reactive subtree fails.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorBoundaryRecovery {
    pub title: String,
    pub summary: String,
    pub details: Vec<String>,
}

impl ErrorBoundaryRecovery {
    /// Builds a structured recovery payload from an iterator of error messages.
    pub fn from_errors<I, S>(errors: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let details: Vec<String> = errors
            .into_iter()
            .map(Into::into)
            .filter(|s| !s.trim().is_empty())
            .collect();
        let summary = if details.is_empty() {
            "An unexpected rendering error occurred. You can reset the view to recover.".to_string()
        } else {
            format!(
                "Encountered {} error{} while rendering this view. Your saved state remains intact.",
                details.len(),
                if details.len() == 1 { "" } else { "s" }
            )
        };
        Self {
            title: "Something went wrong".to_string(),
            summary,
            details,
        }
    }

    /// Alias of [`Self::from_errors`] for callers holding string collections.
    pub fn from_error_strings<I, S>(errors: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self::from_errors(errors)
    }
}

/// Formatted single row in a telemetry summary view.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TelemetrySummaryRow {
    pub label: String,
    pub value: String,
    pub note: String,
}

/// Formats millisecond uptime into compact human-readable form (`42s`, `3m 12s`, `2h 5m`).
pub fn format_uptime_millis(millis: u64) -> String {
    let secs = millis / 1000;
    match secs {
        0..=59 => format!("{secs}s"),
        60..=3599 => format!("{}m {}s", secs / 60, secs % 60),
        _ => format!("{}h {}m", secs / 3600, (secs % 3600) / 60),
    }
}

/// Formats elapsed seconds between an event timestamp and the newest snapshot timestamp.
pub fn format_relative_age_millis(entry_millis: u64, newest_millis: u64) -> String {
    format!("{}s ago", newest_millis.saturating_sub(entry_millis) / 1000)
}

/// Builds the canonical 4 KPI cards (`Uptime`, `Events`, `Warnings`, `Errors`) from
/// [`TelemetryMetrics`].
pub fn telemetry_kpi_cards(metrics: &TelemetryMetrics) -> Vec<KpiCardSpec> {
    vec![
        KpiCardSpec {
            label: "Uptime".to_string(),
            value: format_uptime_millis(metrics.uptime_millis),
            subtext: "since the app started".to_string(),
            accent_color: "#38bdf8".to_string(),
        },
        KpiCardSpec {
            label: "Events".to_string(),
            value: metrics.total_events().to_string(),
            subtext: "logged this session".to_string(),
            accent_color: "#818cf8".to_string(),
        },
        KpiCardSpec {
            label: "Warnings".to_string(),
            value: metrics.warnings.to_string(),
            subtext: "logged this session".to_string(),
            accent_color: "#f59e0b".to_string(),
        },
        KpiCardSpec {
            label: "Errors".to_string(),
            value: metrics.errors.to_string(),
            subtext: "logged this session".to_string(),
            accent_color: if metrics.errors == 0 {
                "#10b981".to_string()
            } else {
                "#ef4444".to_string()
            },
        },
    ]
}

/// Captures KPI cards and tabular summary rows from [`TelemetryMetrics`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TelemetrySummary {
    pub uptime_secs: u64,
    pub uptime_label: String,
    pub total_events: u64,
    pub warnings: u64,
    pub errors: u64,
    pub buffered_entries: usize,
    pub buffer_capacity: usize,
    pub cards: Vec<KpiCardSpec>,
    pub rows: Vec<TelemetrySummaryRow>,
}

impl TelemetrySummary {
    /// Captures a fresh summary directly from `spodeian_telemetry::metrics_snapshot()`.
    pub fn capture() -> Self {
        Self::from_metrics(&metrics_snapshot())
    }

    /// Converts a [`TelemetryMetrics`] snapshot into formatted KPI cards and summary rows.
    pub fn from_metrics(metrics: &TelemetryMetrics) -> Self {
        let uptime_secs = metrics.uptime_millis / 1000;
        let uptime_label = format_uptime_millis(metrics.uptime_millis);
        let cards = telemetry_kpi_cards(metrics);
        let rows = cards
            .iter()
            .map(|c| TelemetrySummaryRow {
                label: c.label.clone(),
                value: c.value.clone(),
                note: c.subtext.clone(),
            })
            .collect();
        Self {
            uptime_secs,
            uptime_label,
            total_events: metrics.total_events(),
            warnings: metrics.warnings,
            errors: metrics.errors,
            buffered_entries: metrics.buffered_entries,
            buffer_capacity: metrics.buffer_capacity,
            cards,
            rows,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn theme_normalization_and_toggle_cycle() {
        assert_eq!(normalize_theme("DARK"), "dark");
        assert_eq!(normalize_theme("  light  "), "light");
        assert_eq!(normalize_theme("high-contrast-dark"), "high-contrast-dark");
        assert_eq!(normalize_theme("unknown"), "dark");

        assert_eq!(toggle_theme_value("dark"), "light");
        assert_eq!(toggle_theme_value("light"), "high-contrast-dark");
        assert_eq!(
            toggle_theme_value("high-contrast-dark"),
            "high-contrast-light"
        );
        assert_eq!(toggle_theme_value("high-contrast-light"), "dark");
    }

    #[test]
    fn modal_tab_wrap_boundaries() {
        assert_eq!(
            should_wrap_modal_tab(true, true, false),
            Some(TabWrapTarget::Last)
        );
        assert_eq!(
            should_wrap_modal_tab(false, false, true),
            Some(TabWrapTarget::First)
        );
        assert_eq!(should_wrap_modal_tab(false, true, false), None);
        assert_eq!(should_wrap_modal_tab(true, false, true), None);
        assert!(!trap_tab_within_element("nonexistent-modal", false));
    }

    #[test]
    fn telemetry_summary_formats_kpis_and_uptime() {
        assert_eq!(format_uptime_millis(999), "0s");
        assert_eq!(format_uptime_millis(61_000), "1m 1s");
        assert_eq!(format_uptime_millis(3_720_000), "1h 2m");
        assert_eq!(format_relative_age_millis(5_000, 17_000), "12s ago");
        assert_eq!(format_relative_age_millis(9_000, 1_000), "0s ago");

        let summary = TelemetrySummary::capture();
        assert_eq!(summary.cards.len(), 4);
        assert_eq!(summary.rows.len(), 4);
        assert_eq!(summary.cards[0].label, "Uptime");
    }

    #[test]
    fn error_boundary_recovery_captures_non_empty_details() {
        let empty = ErrorBoundaryRecovery::from_errors(["", "   "]);
        assert!(empty.details.is_empty());
        assert!(empty.summary.contains("reset the view"));

        let single = ErrorBoundaryRecovery::from_error_strings(["Network timeout", ""]);
        assert_eq!(single.details, vec!["Network timeout"]);
        assert!(single.summary.contains("1 error "));

        let multi = ErrorBoundaryRecovery::from_errors(["Decode error", "Storage full"]);
        assert_eq!(multi.details.len(), 2);
        assert!(multi.summary.contains("2 errors"));
    }

    #[test]
    fn default_shortcuts_are_populated() {
        let shortcuts = default_web_shortcuts();
        assert!(shortcuts.iter().any(|s| s.keys == "Enter"));
        assert!(shortcuts.iter().any(|s| s.keys == "Escape"));
    }
}
