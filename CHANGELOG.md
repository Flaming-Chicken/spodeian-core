# Changelog

All notable changes to `spodeian-core` are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and Semantic Versioning.

## [3.0.0] - 2026-11-14

### Added
- `spodeian-persist`: versioned JSON/RON app-state persistence, backup recovery, and schema migration helpers.
- `spodeian-leptos-ui`: Leptos 0.8 theme/dyslexia DOM bridges, modal focus trapping, telemetry summary formatting, and `ErrorBoundaryRecovery`.
- `spodeian-provenance`: whole-token SPDX/CC `LicenseVerifier`, unified `LedgerRecord`, and deterministic `ATTRIBUTIONS.txt` renderer.
- `spodeian-stats`: normal/Student-t/chi2 CDF & quantiles, OLS/WLS/Ridge/Huber/HAC regressions, Jarque-Bera, ADF, KPSS, Ljung-Box, Hurst, and Engle-Granger cointegration.
- `spodeian-forecast`: Holt linear ETS, `KalmanFilter1D`, `Garch11`, split-conformal intervals, pinball loss, and Gaussian CRPS.
- `spodeian-audio`: pure-Rust WAV PCM 16/24/32f encoder/decoder, linear resampling, acoustic quality metrics, and zero-allocation tanh soft-clipper.
- `sml-runtime`: `SELF_OPTIMIZING_META_INTEGRATOR_WGSL` kernel, `test_meta_proptest_invariants`, and `sml_meta_bench`.
- `sml-py-bridge`: `PySelfOptimizingMeta`, `py_meta_integrate`, and `spodeian_ml_optim.py`.

### Changed
- Separated `AI-Examples` curriculum crates (`sml-candle`, `sml-tools`, curriculum `kernels/`/`weights/`) out of `spodeian-core` into `ai-tests/curriculum/`.
- Salvaged `dir_walk` (`dir2txt`) and `serde_compat` (`u64_i64_compat`) into `spodeian-export`.

### Removed
- Removed deprecated shim crates `spodeian-ml-utils` and `spodeian-lake`.
