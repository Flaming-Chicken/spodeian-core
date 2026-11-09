# `spodeian-ml-utils` Migration Notes

**Stability Tier**: `Experimental (0.x)`  
**Current Version**: `0.1.0`

> [!NOTE]
> As an **Experimental** crate in `spodeian-core`, public APIs in `spodeian-ml-utils` may change between minor versions (`0.1.0` -> `0.2.0`). All breaking modifications, head architecture evolutions, and migration paths will be catalogued in this document. Downstream projects must pin dependencies with caret requirements (`^0.1.0`).

---

## Version 0.1.0 (Initial Experimental Release)

### Core Capabilities
- **SafeTensors Streaming Loader**: Zero-copy ingestion of `.safetensors` binary model weights into Candle tensors (`LoadedModelWeights`), INT8 dequantization, and memory footprint telemetry.
- **Dynamic Multi-Horizon Heads**: Output forecast projections for arbitrary non-locked day horizons (`DynamicMultiHorizonHead`).
- **Monotonic Non-Crossing Quantile Heads**: Cumulative softplus delta formulation (`MonotonicQuantileHead`) mathematically guaranteeing $\hat{q}_{\tau_0} \le \hat{q}_{\tau_1} \le \dots \le \hat{q}_{\tau_{M-1}}$ with pinball loss.
- **Continuous Velocity & Flow Matching**: Continuous velocity vector field predictions (`VelocityFlowHead`) for continuous flow equations and RK4 numerical ODE integration.
- **Continuous-Time Horizon Forecast Heads**: Continuous logarithmic-Fourier temporal basis projection (`ContinuousHorizonHead`) enabling querying alpha for arbitrary fractional/infinite time horizons $t \in \mathbb{R}^+$.
- **Recursive Least Squares (RLS) Adaptation**: Real-time tick adaptation (`OnlineRlsHead`, `OnlineRlsFilter64`) with Sherman-Morrison rank-1 covariance updates and exponential forgetting factor.
- **Sharpness-Aware Minimization (SAM)**: Flat-basin adversarial parameter perturbation and L2 norm gradient clipping (`SharpnessAwareMinimizer`, `SharpnessAwareMinimizer64`).
- **Dual-Speed Test-Time Adaptation**: Real-time coordination of slow structural representation updates and fast head adaptation (`DualSpeedAdaptationManager`).
- **Bit-Perfect Parity Factor Library**: Shared implementations of log returns, rolling volatility, EMA, VWAP, rolling Z-score, temporal masking, Gaussian jitter, and time-warping.
- **Logarithmic Number System (LNS)**: 8-bit Log8 representation and WGSL compute kernels replacing FP multiplications with exponent additions.

### Breaking Changes
- N/A (Initial release).

### Upgrade & Migration Guide
- Downstream projects should declare:
  ```toml
  spodeian-ml-utils = { path = "../spodeian-core/crates/spodeian-ml-utils", version = "^0.1.0" }
  ```
