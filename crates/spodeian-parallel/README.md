# `spodeian-parallel`

[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](../../LICENSE)
[![Status](https://img.shields.io/badge/stability-stable%20(v1.0)-success.svg)]()

**Zero-overhead parallel iteration facade with transparent sequential fallback on WebAssembly.**

---

## Overview

Writing high-performance code across native workstations and WebAssembly typically requires scattering verbose `#[cfg(not(target_arch = "wasm32"))]` annotations everywhere.

`spodeian-parallel` solves this by providing a unified prelude:
- On native targets with `feature = "parallel"`, it re-exports full **Rayon** data-parallel iteration (`rayon::prelude::*`).
- On `wasm32` or when `parallel` is disabled, it provides zero-cost inline wrappers executing standard sequential iterators (`into_par_iter()`, `par_iter()`, `par_chunks()`, `par_sort_unstable()`).

---

## Quickstart

```rust
use spodeian_parallel::prelude::*;

let data = vec![1, 2, 3, 4, 5];
let doubled: Vec<i32> = data.into_par_iter().map(|x| x * 2).collect();
assert_eq!(doubled, vec![2, 4, 6, 8, 10]);
```
