# 🧠 Spodeian ML

[![CI](https://github.com/Flaming-Chicken/spodeian-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/Flaming-Chicken/spodeian-ml/actions)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)
[![Python: 3.14+](https://img.shields.io/badge/Python-3.14%2B-3776AB.svg?logo=python)](https://python.org)
[![PyTorch: 2.14+](https://img.shields.io/badge/PyTorch-2.14%2B-EE4C2C.svg?logo=pytorch)](https://pytorch.org)

**Scientific Deep Learning, Neural State Space Models, Universal Quantization Benchmarks, and Columnar Active Learning Engines**

---

## Architecture & Subsystems

1. **Universal Precision & Quantization Engine (`spodeian_ml.bridge.quantization`)**:
   - Evaluates weights, activations, and KV caches across 12 precisions:
     - IEEE Float32, Float16, Google Brain Float (BFloat16)
     - 8-bit Float (`FP8 E4M3`, `FP8 E5M2`), Symmetric `INT8`, Group-wise `INT4` (AWQ/GPTQ style)
     - NormalFloat4 (`NF4` quantile mapping)
     - Type III Unum Posits (`Posit16`, `Posit8` dynamic tapered accuracy)
     - Base-2 Logarithmic Quantization (`Log8`)
     - BitNet 1.58-bit Ternary (`{-1, 0, +1}`)
     - Binary Neural Network 1-bit (`{-1, +1}`)
   - Built-in Signal-to-Quantization-Noise Ratio (SQNR) and Cosine Similarity diagnostic evaluation.

2. **Self-Supervised Representation Learning (`spodeian_ml.ssl`)**:
   - Masked Autoencoder (`MaskedAutoencoder`) with high-ratio random patch masking.
   - Dual-speed online adaptation, Sharpness-Aware Minimization (SAM), and continuous velocity flow heads.

3. **Data Lake Integration (`spodeian_ml.lake`)**:
   - Zero-copy Apache Arrow streaming loader reading columnar Parquet partitions from `spodeian-lake`.
   - Streaming data iterators, memory-mapped caching, and dataset validation.

4. **Experiment Tracking & Terminal Studio (`spodeian_ml.ops`)**:
   - Comprehensive benchmark harness (`run_quantization_benchmark`).
   - Interactive curses/rich-compatible Terminal User Interface (`spodeian-ml` CLI).

---

## Quickstart

```bash
# Run test suite
uv run pytest

# Run quantization benchmark across all 12 formats
uv run spodeian-ml benchmark --precision all

# Launch interactive TUI
uv run spodeian-ml
```
