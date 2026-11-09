# `spodeian-core` (Facade Crate)

[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](../../LICENSE)
[![Status](https://img.shields.io/badge/stability-stable%20(v1.0)-success.svg)]()

**Umbrella facade re-exporting all Spodeian micro-crates under a unified namespace.**

---

## Overview

The `spodeian-core` facade crate unifies all modular micro-crates in the Spodeian foundation. Enabling the default `full` feature automatically re-exports:

- `spodeian_parallel` as `spodeian_core::parallel`
- `spodeian_telemetry` as `spodeian_core::telemetry`
- `spodeian_tokens` as `spodeian_core::tokens`
- `spodeian_export` as `spodeian_core::export`
- `spodeian_cache` as `spodeian_core::cache`
- `spodeian_web_utils` as `spodeian_core::web_utils`
- `spodeian_ui` as `spodeian_core::ui`
- `spodeian_lake` as `spodeian_core::lake`
- `spodeian_ml_utils` as `spodeian_core::ml_utils`

It also exposes a convenient `prelude` module bringing standard ecosystem types and traits into scope.

---

## Quickstart

```rust
use spodeian_core::prelude::*;

// Access online statistics, device routers, design tokens, and UI layout constraints
let mut stats = OnlineSlidingStats::new(5);
stats.push(10.0);
stats.push(20.0);
assert_eq!(stats.mean(), 15.0);
```
