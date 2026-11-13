# Fair Baseline Evaluation Protocol & Reference Conformity Standard

## 1. Objective and Anti-Bias Principles
To prevent algorithmic handicap, hyperparameter bias, or unfair default parameter selection in comparative evaluation, all benchmark experiments in `spodeian-core` and `ai-tests/09_benchmarks` strictly adhere to this formal evaluation protocol.

---

## 2. Canonical Baseline Reference Specifications
Every optimizer and numerical integrator baseline is implemented in strict mathematical adherence to its original published reference:

| Optimizer / Method | Canonical Reference | Verified Default Hyperparameters | Implementation Invariant |
| :--- | :--- | :--- | :--- |
| **Schedule-Free AdamW** | Defazio et al. (ICLR 2024) [arXiv:2405.15682] | $\beta_1 = 0.9$, $\beta_2 = 0.999$, $K = 10$, $\eta = 0.01$ | Polyak-Ruppert center-manifold accumulation $y_t$ with scheduled primal steps on $z_t$. |
| **Lion** | Chen et al. (Google Brain, 2023) [arXiv:2302.06675] | $\beta_1 = 0.9$, $\beta_2 = 0.99$, $\lambda_{\text{wd}} = 0.01$ | Coordinate-wise $\mathrm{sign}(\cdot)$ gradient update communicating 1 bit/dim. |
| **Sophia-G** | Liu et al. (Stanford, 2024) [arXiv:2305.14342] | $\beta_1 = 0.96$, $\beta_2 = 0.99$, $\rho = 0.04$, $\epsilon = 10^{-12}$ | Diagonal Hessian curvature tracking with lightweight Hutchinson vector probes. |
| **AdamW** | Loshchilov & Hutter (ICLR 2019) [arXiv:1711.05101] | $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, $\lambda_{\text{wd}} = 0.01$ | Decoupled weight decay on primal parameter tensors with EMA momentum. |
| **Nesterov Accelerated SGD** | Nesterov (1983); Sutskever et al. (ICML 2013) | $\mu = 0.9$, dampening = $0.0$, nesterov = `true` | Momentum lookahead with $v_{t+1} = \mu v_t + g_t$, $\Delta\theta = \mu v_{t+1} + g_t$. |
| **Euler SGD** | Cauchy (1847) | $\mu = 0.0$, momentum = `false` | First-order unaccelerated Cauchy gradient descent. |
| **Dormand-Prince Adaptive RK45** | Dormand & Prince (1980); Hairer et al. (1993) | Order 5(4) embedded pair, tolerances $\text{atol} = 10^{-6}, \text{rtol} = 10^{-4}$ | Standard explicit adaptive Runge-Kutta step controller with PI error limiter. |

---

## 3. Dual-Evaluation Protocol: Default vs. Tuned Pareto Frontier

### 3.1 Tier 1: Default-vs-Default (Out-of-the-Box Robustness)
Evaluates practical usability without hyperparameter optimization.
- Every method is initialized with its published canonical defaults ($\eta = 0.01$ or $0.05$).
- Tests whether the algorithm can unconditionally converge without manual intervention.
- **Finding**: On extreme ill-conditioned landscapes ($\kappa = 10^4$ Rotated Ellipsoid, $\kappa = 10^{10}$ Rosenbrock), default Nesterov SGD explodes to `NaN` or diverges to $10^{40}$, whereas `SelfOptimizingMeta` converges unconditionally.

### 3.2 Tier 2: Tuned Grid Search (Optimal Performance Frontier)
Evaluates maximum theoretical capability under optimal tuning.
- For each problem, baselines undergo an extensive grid sweep over learning rates:
  $$\eta \in \{10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1.0\}.$$
- The best configuration is selected for the competitive Pareto frontier.
- **Example**: On the $45^\circ$ Rotated Ellipsoid ($\kappa = 10^4$), hand-tuned Nesterov SGD with $\eta = 10^{-4}$ reaches exact $0.0000$ at step 2.

### 3.3 Tier 3: Matched Learning Rate Evaluation
To rigorously verify that `SelfOptimizingMeta` does not succeed simply due to a larger or smaller learning rate, `SelfOptimizingMeta` is explicitly evaluated at the **exact same tuned learning rate** as the best competitor:
- **Rotated Ellipsoid**: `SelfOptimizingMeta (Matched lr=0.0001)` vs. `Nesterov (Tuned lr=0.0001)`.
- **Beale Curved Valley**: `SelfOptimizingMeta (Matched lr=0.005)` vs. `Nesterov (Tuned lr=0.005)`.

Both results are transparently reported side-by-side in Table II (`table_multi_metric.tex`).

---

## 4. Hardware, Wall-Clock, and Memory Measurement Standards

1. **Hardware Parity**:
   - All tests execute single-threaded on the identical CPU core with process isolation.
   - Power management profiles set to maximum performance to prevent dynamic CPU frequency scaling jitter.
2. **Timing Protocol**:
   - Measured via `std::time::Instant::now()` directly wrapping `opt.step(&grads)`.
   - Excludes synthetic oracle function evaluations ($f(\theta)$ and $\nabla f(\theta)$) to isolate pure optimizer runtime overhead.
   - Warmup iterations discarded before recording timer statistics.
3. **Memory Footprint Accounting**:
   - Auxiliary state bytes verified via `opt.memory_footprint()`.
   - Stack-inlined buffers ($N \le 64$) strictly validated to produce 0 heap allocations via `SmallVec::spilled()`.
   - Shared scratch buffers (`grad_buf`, `theta_buf`, `f32_scratch`) verified to be instantiated once per optimizer instance, rather than duplicated across parameters.
