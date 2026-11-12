# `spodeian-export`

[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](../../LICENSE)
[![Status](https://img.shields.io/badge/stability-stable%20(v2.0)-success.svg)]()

**Universal serialization, zero-copy CSV streaming, compressed BSON, SafeTensors, and WAV audio.**

---

## Overview

`spodeian-export` handles high-throughput serialization and multi-format dataset exports:

- **RFC 4180 CSV / TSV**: Escaping and streaming parsing with zero-copy borrowed string views (`parse_csv_line_borrowed`).
- **Compressed BSON**: High-density binary serialization compressed with Zlib/miniz_oxide.
- **Audio WAV (PCM-16 & Float-32)**: Pure-Rust RIFF/WAVE audio serialization for CD-quality playback across browsers and native media players.
- **SafeTensors Serialization**: Checkpointing multi-tensor weight dictionaries with zero-copy memory mapping.

---

## Quickstart

```rust
use spodeian_export::{escape_csv_field, parse_csv_line};

// Escape text for CSV
let escaped = escape_csv_field("Hello, \"World\"");
assert_eq!(escaped, "\"Hello, \"\"World\"\"\"");

// Parse a CSV line into discrete fields
let fields = parse_csv_line("AAPL,150.25,\"New York, NY\"");
assert_eq!(fields, vec!["AAPL", "150.25", "New York, NY"]);
```
