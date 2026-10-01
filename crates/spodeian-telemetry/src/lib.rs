//! Universal telemetry, logging, in-memory diagnostic buffer, and panic capture system.
//!
//! **Stability: Stable** — semver-guaranteed since v1.0
//!
//! Provides identical structured JSON or ANSI console logs on desktop and mobile,
//! exposes a thread-safe circular `LogRingBuffer` for in-app GUI log viewers,
//! installs panic diagnostic hooks, and directs output to the browser developer console on WASM.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::{Arc, OnceLock, RwLock};

static GLOBAL_BUFFER: OnceLock<Arc<LogRingBuffer>> = OnceLock::new();

/// Structured record representing an intercepted tracing event.
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogEntry {
    /// Timestamp in milliseconds since UNIX epoch.
    pub timestamp_millis: u64,
    /// Tracing severity level string ("ERROR", "WARN", "INFO", "DEBUG", "TRACE").
    pub level: String,
    /// Tracing target or module path.
    pub target: String,
    /// Primary log message.
    pub message: String,
    /// Structured contextual key-value fields.
    pub fields: Vec<(String, String)>,
}

/// Thread-safe circular in-memory buffer storing recent log entries for diagnostic UI viewers.
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
#[derive(Debug)]
pub struct LogRingBuffer {
    capacity: usize,
    entries: RwLock<VecDeque<LogEntry>>,
}

impl LogRingBuffer {
    /// Creates a new circular ring buffer with the given capacity.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            entries: RwLock::new(VecDeque::with_capacity(capacity.min(500))),
        }
    }

    /// Appends a log entry to the buffer, evicting the oldest entry if capacity is reached.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
    pub fn push(&self, entry: LogEntry) {
        if let Ok(mut entries) = self.entries.write() {
            if entries.len() >= self.capacity {
                entries.pop_front();
            }
            entries.push_back(entry);
        }
    }

    /// Returns a cloned snapshot of all entries currently stored in the buffer.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
    pub fn snapshot(&self) -> Vec<LogEntry> {
        self.entries
            .read()
            .map(|entries| entries.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Returns the number of entries currently stored.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
    pub fn len(&self) -> usize {
        self.entries.read().map(|e| e.len()).unwrap_or(0)
    }

    /// Returns true if the buffer contains zero entries.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clears all stored log entries.
    #[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
    pub fn clear(&self) {
        if let Ok(mut entries) = self.entries.write() {
            entries.clear();
        }
    }
}

/// Returns the shared global in-memory log ring buffer.
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
pub fn global_ring_buffer() -> Arc<LogRingBuffer> {
    GLOBAL_BUFFER
        .get_or_init(|| Arc::new(LogRingBuffer::new(2000)))
        .clone()
}

fn current_time_millis() -> u64 {
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }
    #[cfg(target_arch = "wasm32")]
    {
        0
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct RingBufferLayer {
    buffer: Arc<LogRingBuffer>,
}

#[cfg(not(target_arch = "wasm32"))]
struct FieldVisitor {
    message: String,
    fields: Vec<(String, String)>,
}

#[cfg(not(target_arch = "wasm32"))]
impl tracing::field::Visit for FieldVisitor {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            let s = format!("{value:?}");
            self.message = if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
                s[1..s.len() - 1].to_string()
            } else {
                s
            };
        } else {
            self.fields
                .push((field.name().to_string(), format!("{value:?}")));
        }
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        if field.name() == "message" {
            self.message = value.to_string();
        } else {
            self.fields
                .push((field.name().to_string(), value.to_string()));
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for RingBufferLayer {
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: tracing_subscriber::layer::Context<'_, S>) {
        let mut visitor = FieldVisitor {
            message: String::new(),
            fields: Vec::new(),
        };
        event.record(&mut visitor);

        let entry = LogEntry {
            timestamp_millis: current_time_millis(),
            level: event.metadata().level().to_string(),
            target: event.metadata().target().to_string(),
            message: visitor.message,
            fields: visitor.fields,
        };
        self.buffer.push(entry);
    }
}

/// Installs a crash diagnostic panic hook.
///
/// On desktop and mobile, catches panics, prints a formatted error message, and
/// records a crash log. On WebAssembly, installs `console_error_panic_hook`.
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
pub fn install_panic_hook(app_name: &'static str) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::panic::set_hook(Box::new(move |panic_info| {
            let timestamp = current_time_millis();
            let thread = std::thread::current();
            let thread_name = thread.name().unwrap_or("unnamed");

            let location = panic_info
                .location()
                .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
                .unwrap_or_else(|| "unknown location".to_string());

            let msg = if let Some(s) = panic_info.payload().downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = panic_info.payload().downcast_ref::<String>() {
                s.clone()
            } else {
                "Unknown panic payload".to_string()
            };

            let diagnostic = format!(
                "=== CRASH REPORT [{app_name}] ===\nTimestamp: {timestamp}\nThread: {thread_name}\nLocation: {location}\nMessage: {msg}\n"
            );

            eprintln!("{diagnostic}");

            // Attempt to write crash dump to current directory
            let filename = format!("crash_{app_name}_{timestamp}.log");
            let _ = std::fs::write(&filename, &diagnostic);
        }));
    }

    #[cfg(target_arch = "wasm32")]
    {
        let _ = app_name;
        console_error_panic_hook::set_once();
    }
}

/// Configures and initializes global tracing subscriber, panic handlers, and in-memory log buffer.
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
pub fn init_telemetry_with_buffer(app_name: &'static str, default_filter: &str) -> Arc<LogRingBuffer> {
    let buffer = global_ring_buffer();
    install_panic_hook(app_name);

    #[cfg(not(target_arch = "wasm32"))]
    {
        use tracing_subscriber::{fmt, prelude::*, EnvFilter};
        let filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new(default_filter));

        let ring_layer = RingBufferLayer {
            buffer: buffer.clone(),
        };

        let _ = tracing_subscriber::registry()
            .with(fmt::layer())
            .with(ring_layer)
            .with(filter)
            .try_init();
    }

    #[cfg(target_arch = "wasm32")]
    {
        let _ = tracing_subscriber::fmt()
            .with_writer(tracing_web::MakeConsoleWriter)
            .without_time()
            .with_ansi(false)
            .try_init();
    }

    buffer
}

/// Configures and initializes global tracing subscriber and panic handlers.
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
pub fn init_telemetry(default_filter: &str) {
    let _ = init_telemetry_with_buffer("spodeian_app", default_filter);
}

/// Convenience initializer using standard desktop/egui filter:
/// `"info,wgpu=warn,egui=warn"`
#[doc = "**Stability: Stable** — semver-guaranteed since v1.0"]
pub fn init_default() {
    init_telemetry("info,wgpu=warn,egui=warn");
}
