# `spodeian-telemetry`

[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](../../LICENSE)
[![Status](https://img.shields.io/badge/stability-stable%20(v1.0)-success.svg)]()

**Universal zero-overhead tracing, in-memory diagnostic ring buffer, and crash report panic hooks.**

---

## Overview

`spodeian-telemetry` provides structured observability across desktop, WASM web, and embedded contexts:

- **Structured Tracing**: Native support for `tracing`, `tracing-subscriber`, and `tracing-web`.
- **In-Memory Ring Buffer (`LogRingBuffer`)**: Zero-allocation thread-safe log sink with zero-copy visitors (`for_each_recent`) designed for 60 FPS egui diagnostic consoles.
- **Panic Diagnostic Hooks**: Intercepts panics, captures backtraces and memory statistics, and logs structured diagnostic reports before crash.
- **Web Console Forwarding**: On WASM, automatically directs logs to browser developer tools with color-coded severity.

---

## Quickstart

```rust
use spodeian_telemetry::{LogRingBuffer, LogEntry};

// Initialize an in-memory buffer holding up to 500 recent entries
let buffer = LogRingBuffer::new(500);

buffer.push(LogEntry {
    timestamp_millis: 1700000000000,
    level: "INFO".to_string(),
    target: "engine::solver".to_string(),
    message: "Convergence reached in 45 iterations".to_string(),
    fields: vec![("loss".to_string(), "1e-16".to_string())],
});

assert_eq!(buffer.len(), 1);
```
