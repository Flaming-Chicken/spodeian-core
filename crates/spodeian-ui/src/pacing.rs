//! Reactive frame pacer for egui desktop and web applications.
//!
//! **Stability: Stable** — semver-guaranteed since v1.0
//!
//! Intelligently throttles redraw loops to zero or low framerates when the application
//! is idle, conserving battery and reducing GPU/CPU power consumption.

use egui::Context;
use std::time::{Duration, Instant};

/// Reactive repaint manager that paces UI redraws based on user interactions and animations.
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
#[derive(Debug)]
pub struct ReactiveFramePacer {
    /// Timestamp of the last detected user or programmatic activity.
    last_activity: Instant,
    /// Duration of high-refresh responsiveness after the last event before entering idle mode.
    active_linger: Duration,
    /// Maximum time between background redraws when fully idle.
    idle_repaint_interval: Duration,
}

impl Default for ReactiveFramePacer {
    fn default() -> Self {
        Self::new(Duration::from_millis(500), Duration::from_secs(1))
    }
}

impl ReactiveFramePacer {
    /// Creates a new frame pacer with explicit active linger and idle intervals.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
    pub fn new(active_linger: Duration, idle_repaint_interval: Duration) -> Self {
        Self {
            last_activity: Instant::now(),
            active_linger,
            idle_repaint_interval,
        }
    }

    /// Explicitly signals activity (e.g., streaming network update, tick event).
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
    pub fn notify_activity(&mut self) {
        self.last_activity = Instant::now();
    }

    /// Inspects the context input and schedules the appropriate next repaint.
    ///
    /// If user input (pointer movement, key press) occurred or `is_animating` is true,
    /// marks active and requests an immediate repaint. Otherwise, schedules a throttled
    /// idle repaint after `idle_repaint_interval`.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
    pub fn pace_frame(&mut self, ctx: &Context, is_animating: bool) {
        let has_input = ctx.input(|i| {
            !i.raw.events.is_empty()
                || i.pointer.is_moving()
                || i.pointer.any_down()
                || i.raw.focused
        });

        if has_input || is_animating {
            self.notify_activity();
        }

        let elapsed = self.last_activity.elapsed();
        if elapsed < self.active_linger || is_animating {
            // Actively interacting: repaint as fast as display refreshes
            ctx.request_repaint();
        } else {
            // Idle state: request repaint after scheduled delay
            ctx.request_repaint_after(self.idle_repaint_interval);
        }
    }

    /// Returns true if the pacer is currently in idle low-power mode.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
    pub fn is_idle(&self) -> bool {
        self.last_activity.elapsed() >= self.active_linger
    }
}
