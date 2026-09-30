# `spodeian-py-bridge` Migration Notes

**Stability Tier**: `Experimental (0.x)`  
**Current Version**: `0.1.0`

> [!NOTE]
> As an **Experimental** crate in `spodeian-core`, public FFI symbols and exported PyO3 functions in `spodeian-py-bridge` may change between minor versions (`0.1.0` -> `0.2.0`). All breaking modifications and migration steps will be catalogued here. Downstream projects must pin dependencies with caret requirements (`^0.1.0`).

---

## Version 0.1.0 (Initial Experimental Release)

### Core Capabilities
- **PyO3 Extension Module**: Native Rust kernels compiled to CPython extension (`spodeian_py_bridge`) with `abi3-py310` stable ABI compatibility.
- **Fast Data Augmentations**:
  - `py_temporal_mask`: Deterministic pseudo-random sequence masking for self-supervised learning.
  - `py_gaussian_jitter`: Box-Muller Gaussian jitter injection.
  - `py_rolling_zscore`: In-place sliding window normalisation.
  - `py_time_warp`: Linear interpolation resampling for variable sequence lengths.
- **Financial Kernels**:
  - `py_log_returns`: Logarithmic returns calculation.
  - `py_rolling_volatility`: Rolling sample standard deviation annualized by $\sqrt{252}$.
  - `py_ema`: Exponential moving average.
  - `py_vwap`: Cumulative volume-weighted average price.

### Breaking Changes
- N/A (Initial release).

### Upgrade & Migration Guide
- Downstream projects should declare:
  ```toml
  spodeian-py-bridge = { path = "../spodeian-core/crates/spodeian-py-bridge", version = "^0.1.0" }
  ```
