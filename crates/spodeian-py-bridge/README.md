# `spodeian-py-bridge`

[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](../../LICENSE)
[![Status](https://img.shields.io/badge/stability-beta%20(v0.1)-yellow.svg)]()

**High-throughput zero-copy Python bridge via PyO3, Arrow arrays, and rolling indicators.**

---

## Overview

`spodeian-py-bridge` provides high-performance CPython FFI bindings allowing Python runtimes (`spodeian-ml`, research notebooks) to invoke native Rust operations with zero serialization overhead:

- **PyO3 Extension Module**: Compiled as a native Python extension module supporting Python 3.10+.
- **Zero-Copy Arrow Interchange**: Interchanges columnar Arrow tables and PyArrow record batches directly with `spodeian-lake`.
- **Accelerated Augmentations**: Real-time signal augmentations and rolling statistical indicators compiled into native SIMD machine code.
