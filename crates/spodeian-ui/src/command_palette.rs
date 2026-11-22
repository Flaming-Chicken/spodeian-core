//! Searchable keyboard-navigable command palette modal widget for egui applications.

use egui::{Context, Key, RichText, ScrollArea, TextEdit};

/// Single executable entry displayed inside [`CommandPaletteWidget`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandEntry {
    pub id: String,
    pub label: String,
    pub shortcut_hint: String,
    pub category: String,
}

impl CommandEntry {
    /// Creates a command entry with an identifier, display label, optional shortcut hint, and category.
    pub fn new(
        id: impl Into<String>,
        label: impl Into<String>,
        shortcut_hint: impl Into<String>,
        category: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            shortcut_hint: shortcut_hint.into(),
            category: category.into(),
        }
    }
}

/// Interactive searchable command palette modal supporting Up/Down/Enter/Escape navigation.
#[derive(Debug, Clone, Default)]
pub struct CommandPaletteWidget {
    open: bool,
    query: String,
    selected_index: usize,
    commands: Vec<CommandEntry>,
    focus_search_on_open: bool,
}

impl CommandPaletteWidget {
    /// Creates a new command palette pre-populated with `commands`.
    pub fn new(commands: Vec<CommandEntry>) -> Self {
        Self {
            open: false,
            query: String::new(),
            selected_index: 0,
            commands,
            focus_search_on_open: false,
        }
    }

    /// Opens the palette and focuses the search input on the next frame.
    pub fn open(&mut self) {
        self.open = true;
        self.selected_index = 0;
        self.focus_search_on_open = true;
    }

    /// Closes the palette and resets the search filter.
    pub fn close(&mut self) {
        self.open = false;
        self.query.clear();
        self.selected_index = 0;
        self.focus_search_on_open = false;
    }

    /// Returns whether the palette modal is currently visible.
    pub fn is_open(&self) -> bool {
        self.open
    }

    /// Updates open visibility cleanly.
    pub fn set_open(&mut self, open: bool) {
        if open {
            self.open();
        } else {
            self.close();
        }
    }

    /// Replaces the command catalog while clamping the active selection index.
    pub fn set_commands(&mut self, commands: Vec<CommandEntry>) {
        self.commands = commands;
        self.selected_index = 0;
    }

    /// Returns the current search query string.
    pub fn query(&self) -> &str {
        &self.query
    }

    /// Sets the search query (useful in tests or external triggers) and clamps selection.
    pub fn set_query(&mut self, query: impl Into<String>) {
        self.query = query.into();
        self.selected_index = 0;
    }

    /// Returns the currently highlighted index within the filtered result list.
    pub fn selected_index(&self) -> usize {
        self.selected_index
    }

    /// Returns indices into the command list matching the current query (case-insensitive).
    pub fn filtered_indices(&self) -> Vec<usize> {
        let needle = self.query.trim().to_lowercase();
        if needle.is_empty() {
            return (0..self.commands.len()).collect();
        }

        self.commands
            .iter()
            .enumerate()
            .filter(|(_, entry)| {
                entry.label.to_lowercase().contains(&needle)
                    || entry.category.to_lowercase().contains(&needle)
                    || entry.id.to_lowercase().contains(&needle)
                    || entry.shortcut_hint.to_lowercase().contains(&needle)
            })
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Renders the command palette when open and returns `Some(command_id)` if a command was
    /// executed by keyboard (`Enter`) or pointer click.
    pub fn show(&mut self, ctx: &Context) -> Option<String> {
        if !self.open {
            return None;
        }

        let escape_pressed = ctx.input(|i| i.key_pressed(Key::Escape));
        if escape_pressed {
            self.close();
            return None;
        }

        let filtered = self.filtered_indices();
        if filtered.is_empty() {
            self.selected_index = 0;
        } else if self.selected_index >= filtered.len() {
            self.selected_index = filtered.len() - 1;
        }

        let down_pressed = ctx.input(|i| i.key_pressed(Key::ArrowDown));
        let up_pressed = ctx.input(|i| i.key_pressed(Key::ArrowUp));
        let enter_pressed = ctx.input(|i| i.key_pressed(Key::Enter));

        if !filtered.is_empty() {
            if down_pressed {
                self.selected_index = (self.selected_index + 1) % filtered.len();
            } else if up_pressed {
                self.selected_index = if self.selected_index == 0 {
                    filtered.len() - 1
                } else {
                    self.selected_index - 1
                };
            }
        }

        let mut executed_id: Option<String> = None;
        if enter_pressed && !filtered.is_empty() {
            let cmd_idx = filtered[self.selected_index];
            executed_id = Some(self.commands[cmd_idx].id.clone());
            self.close();
            return executed_id;
        }

        let mut window_open = self.open;
        egui::Window::new("Command Palette")
            .open(&mut window_open)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_TOP, egui::vec2(0.0, 64.0))
            .fixed_size(egui::vec2(520.0, 320.0))
            .show(ctx, |ui| {
                let edit = TextEdit::singleline(&mut self.query)
                    .hint_text("Type a command, category, or shortcut...")
                    .desired_width(f32::INFINITY);
                let resp = ui.add(edit);
                if self.focus_search_on_open {
                    resp.request_focus();
                    self.focus_search_on_open = false;
                }
                if resp.changed() {
                    self.selected_index = 0;
                }

                ui.add_space(6.0);
                ui.separator();

                let filtered_now = self.filtered_indices();
                ScrollArea::vertical().max_height(240.0).show(ui, |ui| {
                    if filtered_now.is_empty() {
                        ui.weak("No matching commands.");
                        return;
                    }

                    for (visible_idx, cmd_idx) in filtered_now.iter().enumerate() {
                        let cmd = &self.commands[*cmd_idx];
                        let selected = visible_idx == self.selected_index;
                        let clicked = ui
                            .horizontal(|ui| {
                                let label = format!("[{}] {}", cmd.category, cmd.label);
                                let resp = ui.selectable_label(selected, label);
                                if !cmd.shortcut_hint.is_empty() {
                                    ui.with_layout(
                                        egui::Layout::right_to_left(egui::Align::Center),
                                        |ui| {
                                            ui.label(
                                                RichText::new(&cmd.shortcut_hint)
                                                    .weak()
                                                    .monospace(),
                                            );
                                        },
                                    );
                                }
                                resp.clicked()
                            })
                            .inner;

                        if clicked {
                            executed_id = Some(cmd.id.clone());
                        }
                    }
                });
            });

        if !window_open || executed_id.is_some() {
            self.close();
        }

        executed_id
    }
}

/// Normalizes a keyboard shortcut hint (such as `"ctrl + shift + k"`) into a canonical form
/// (`"Ctrl+Shift+K"`) with modifiers ordered `Ctrl`, `Shift`, `Alt`, `Meta`.
pub fn normalize_shortcut_hint(hint: &str) -> String {
    let trimmed = hint.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    let mut has_ctrl = false;
    let mut has_shift = false;
    let mut has_alt = false;
    let mut has_meta = false;
    let mut key_part: Option<String> = None;

    for raw_part in trimmed.split('+') {
        let token = raw_part.trim();
        if token.is_empty() {
            continue;
        }
        match token.to_ascii_lowercase().as_str() {
            "ctrl" | "control" => has_ctrl = true,
            "shift" => has_shift = true,
            "alt" | "option" => has_alt = true,
            "meta" | "cmd" | "command" | "super" | "win" => has_meta = true,
            "esc" | "escape" => key_part = Some("Esc".to_string()),
            "enter" | "return" => key_part = Some("Enter".to_string()),
            "space" | "spacebar" => key_part = Some("Space".to_string()),
            "del" | "delete" => key_part = Some("Delete".to_string()),
            other => {
                if other.len() == 1 {
                    key_part = Some(other.to_ascii_uppercase());
                } else if let Some(rest) = other.strip_prefix('f')
                    && !rest.is_empty()
                    && rest.chars().all(|c| c.is_ascii_digit())
                {
                    key_part = Some(format!("F{rest}"));
                } else {
                    key_part = Some(token.to_string());
                }
            }
        }
    }

    let mut parts = Vec::with_capacity(5);
    if has_ctrl {
        parts.push("Ctrl".to_string());
    }
    if has_shift {
        parts.push("Shift".to_string());
    }
    if has_alt {
        parts.push("Alt".to_string());
    }
    if has_meta {
        parts.push("Meta".to_string());
    }
    if let Some(key) = key_part {
        parts.push(key);
    }

    parts.join("+")
}

/// Finds duplicate normalized shortcut hints across `entries`, returning sorted `(shortcut, command_ids)` pairs.
pub fn detect_shortcut_conflicts(entries: &[CommandEntry]) -> Vec<(String, Vec<String>)> {
    let mut by_shortcut: std::collections::BTreeMap<String, Vec<String>> =
        std::collections::BTreeMap::new();

    for entry in entries {
        let normalized = normalize_shortcut_hint(&entry.shortcut_hint);
        if normalized.is_empty() {
            continue;
        }
        by_shortcut
            .entry(normalized)
            .or_default()
            .push(entry.id.clone());
    }

    by_shortcut
        .into_iter()
        .filter(|(_, ids)| ids.len() > 1)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shortcut_normalization_and_conflict_detection() {
        assert_eq!(normalize_shortcut_hint("  ctrl + k "), "Ctrl+K");
        assert_eq!(normalize_shortcut_hint("Shift+Ctrl+l"), "Ctrl+Shift+L");
        assert_eq!(normalize_shortcut_hint("f11"), "F11");
        assert_eq!(normalize_shortcut_hint(""), "");

        let entries = vec![
            CommandEntry::new("cmd.palette", "Command Palette", "Ctrl+K", "Navigation"),
            CommandEntry::new("cmd.search", "Search", "k + ctrl", "Navigation"),
            CommandEntry::new("cmd.help", "Help", "F1", "Help"),
            CommandEntry::new("cmd.none", "No Shortcut", "", "General"),
        ];

        let conflicts = detect_shortcut_conflicts(&entries);
        assert_eq!(
            conflicts,
            vec![(
                "Ctrl+K".to_string(),
                vec!["cmd.palette".to_string(), "cmd.search".to_string()]
            )]
        );
    }
}
