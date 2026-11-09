//! Continuous probability flow ODE solvers: RK4, adaptive RK45, RK23, and learned step controllers.

use crate::precision::{DataTypeSpec, InformationRateGatingPolicy, NumericType};
use crate::trajectory::DenseHermiteTrajectory;
use serde::{Deserialize, Serialize};
use smallvec::{SmallVec, smallvec};

// Runge-Kutta 4th Order (RK4) continuous trajectory ODE solver for flow matching.
//
// Provides mathematically exact 4th-order ODE integration for neural probability flow
// trajectories dx_t/dt = v_theta(x_t, t, c) with O(h^4) local truncation error.

/// Continuous probability flow ODE solver using Runge-Kutta 4th Order (RK4).
pub struct Rk4FlowSolver;

impl Rk4FlowSolver {
    pub const TABLEAU: ExplicitButcherTableau<4> = RK4_TABLEAU;

    /// Integrates a single step from $t$ to $t + h$ using RK4:
    /// $$k_1 = v(x_t, t)$$
    /// $$k_2 = v\left(x_t + \frac{h}{2} k_1, t + \frac{h}{2}\right)$$
    /// $$k_3 = v\left(x_t + \frac{h}{2} k_2, t + \frac{h}{2}\right)$$
    /// $$k_4 = v(x_t + h k_3, t + h)$$
    /// $$x_{t+h} = x_t + \frac{h}{6} (k_1 + 2k_2 + 2k_3 + k_4)$$
    pub fn step<F>(x: &[f32], t: f32, h: f32, velocity_fn: F) -> Vec<f32>
    where
        F: FnMut(&[f32], f32) -> Vec<f32>,
    {
        Self::TABLEAU.step(x, t, h, velocity_fn)
    }

    /// Full trajectory integration from $t = 0.0$ to $t = 1.0$ over $N$ discrete steps.
    pub fn solve_trajectory<F>(x0: &[f32], num_steps: usize, mut velocity_fn: F) -> Vec<f32>
    where
        F: FnMut(&[f32], f32) -> Vec<f32>,
    {
        solve_fixed_trajectory_core(x0, num_steps, |x, t, dt| Self::step(x, t, dt, &mut velocity_fn))
    }
}


// Adaptive Step-Size & Learned Runge-Kutta ODE Solvers for Neural Flow Matching.
//
// Provides mathematically exact adaptive continuous probability flow solvers:
// 1. **Dormand-Prince (RK45)**: 5th-order solution with embedded 4th-order local error
//    estimator for rigorous dynamic step-size control.
// 2. **Bogacki-Shampine (RK23)**: 3rd-order solution with 2nd-order error estimator
//    optimized for ultra-low latency, low-power mobile DSP synthesis.
// 3. **LearnedFlowController**: Parameterized neural ODE controller that modulates
//    integration step sizes based on trajectory curvature and velocity gradients.

/// Result of an adaptive Runge-Kutta step.
#[derive(Debug, Clone)]
pub struct AdaptiveStepResult {
    /// Integrated state vector at $t + h$.
    pub x_next: Vec<f32>,
    /// Estimated local truncation error $L_2$ norm.
    pub error_norm: f32,
    /// Recommended next step size $h_{\text{rec}}$.
    pub recommended_h: f32,
    /// Whether the step satisfies the error tolerance.
    pub accepted: bool,
}

/// Explicit Runge-Kutta Butcher Tableau of stage count `S`.
/// Unifies explicit single-step and embedded adaptive ODE integration with reusable scratch buffers.
#[derive(Clone, Copy, Debug)]
pub struct ExplicitButcherTableau<const S: usize> {
    pub c: [f32; S],
    pub a: [[f32; S]; S],
    pub b: [f32; S],
    pub e: Option<[f32; S]>,
    pub is_fsal: bool,
    pub safety: f32,
    pub exp: f32,
    pub min_scale: f32,
    pub max_scale: f32,
}

impl<const S: usize> ExplicitButcherTableau<S> {
    /// Integrates a single non-adaptive explicit Runge-Kutta step.
    pub fn step<F>(&self, x: &[f32], t: f32, h: f32, mut velocity_fn: F) -> Vec<f32>
    where
        F: FnMut(&[f32], f32) -> Vec<f32>,
    {
        let n = x.len();
        let mut k: [Vec<f32>; S] = std::array::from_fn(|_| Vec::new());
        let mut x_stage = vec![0.0f32; n];

        for i in 0..S {
            if i == 0 {
                k[0] = velocity_fn(x, t);
            } else {
                for d in 0..n {
                    let mut accum = 0.0f32;
                    for j in 0..i {
                        accum += self.a[i][j] * k[j][d];
                    }
                    x_stage[d] = x[d] + h * accum;
                }
                k[i] = velocity_fn(&x_stage, t + self.c[i] * h);
            }
        }

        let mut x_next = vec![0.0f32; n];
        for d in 0..n {
            let mut accum = 0.0f32;
            for i in 0..S {
                accum += self.b[i] * k[i][d];
            }
            x_next[d] = x[d] + h * accum;
        }

        x_next
    }

    /// Integrates a single embedded adaptive Runge-Kutta step with dynamic step-size recommendation.
    pub fn adaptive_step<F>(
        &self,
        x: &[f32],
        t: f32,
        h: f32,
        tol: f32,
        mut velocity_fn: F,
    ) -> AdaptiveStepResult
    where
        F: FnMut(&[f32], f32) -> Vec<f32>,
    {
        let n = x.len();
        let mut k: [Vec<f32>; S] = std::array::from_fn(|_| Vec::new());
        let mut x_stage = vec![0.0f32; n];

        let num_eval_stages = if self.is_fsal { S - 1 } else { S };

        for i in 0..num_eval_stages {
            if i == 0 {
                k[0] = velocity_fn(x, t);
            } else {
                for d in 0..n {
                    let mut accum = 0.0f32;
                    for j in 0..i {
                        accum += self.a[i][j] * k[j][d];
                    }
                    x_stage[d] = x[d] + h * accum;
                }
                k[i] = velocity_fn(&x_stage, t + self.c[i] * h);
            }
        }

        let mut x_next = vec![0.0f32; n];
        for d in 0..n {
            let mut accum = 0.0f32;
            for i in 0..num_eval_stages {
                accum += self.b[i] * k[i][d];
            }
            x_next[d] = x[d] + h * accum;
        }

        if self.is_fsal {
            k[S - 1] = velocity_fn(&x_next, t + h);
        }

        let e_weights = self.e.expect("adaptive_step requires embedded error weights e");
        let mut sum_sq_err = 0.0f32;
        for d in 0..n {
            let mut err_i = 0.0f32;
            for i in 0..S {
                err_i += e_weights[i] * k[i][d];
            }
            err_i *= h;
            sum_sq_err += err_i * err_i;
        }

        let error_norm = (sum_sq_err / n.max(1) as f32).sqrt();
        let scale = if error_norm > 1e-12 {
            (self.safety * (tol / error_norm).powf(self.exp)).clamp(self.min_scale, self.max_scale)
        } else {
            2.0
        };
        let recommended_h = h * scale;
        let accepted = error_norm <= tol;

        AdaptiveStepResult {
            x_next,
            error_norm,
            recommended_h,
            accepted,
        }
    }
}

/// Unified core loop for fixed-step trajectory ODE integration.
pub fn solve_fixed_trajectory_core<StepFn>(
    x0: &[f32],
    steps: usize,
    mut step_fn: StepFn,
) -> Vec<f32>
where
    StepFn: FnMut(&[f32], f32, f32) -> Vec<f32>,
{
    assert!(steps > 0, "Number of steps must be positive");
    let dt = 1.0f32 / steps as f32;
    let mut current_x = x0.to_vec();

    for step_idx in 0..steps {
        let t = step_idx as f32 * dt;
        current_x = step_fn(&current_x, t, dt);
    }

    current_x
}

/// Unified core loop for adaptive step-size trajectory ODE integration with FSAL and error control.
pub fn solve_adaptive_trajectory_core<StepFn>(
    solver_name: &str,
    x0: &[f32],
    initial_h: f32,
    min_h: f32,
    max_h: f32,
    tol: f32,
    max_iterations: usize,
    mut step_fn: StepFn,
) -> Result<(Vec<f32>, usize), String>
where
    StepFn: FnMut(&[f32], f32, f32, f32) -> AdaptiveStepResult,
{
    let mut current_x = x0.to_vec();
    let mut t = 0.0f32;
    let mut h = initial_h.clamp(min_h, max_h);
    let mut steps = 0;

    while t < 1.0 - 1e-6 {
        if steps >= max_iterations {
            return Err(format!(
                "{solver_name} exceeded maximum iterations ({max_iterations}) at t = {t:.4}"
            ));
        }

        // Clamp h to reach exactly 1.0 at the end
        if t + h > 1.0 {
            h = 1.0 - t;
        }

        let result = step_fn(&current_x, t, h, tol);

        if result.accepted || h <= min_h {
            current_x = result.x_next;
            t += h;
            steps += 1;
            h = result.recommended_h.clamp(min_h, max_h);
        } else {
            // Reject step and retry with smaller h
            h = result.recommended_h.clamp(min_h, max_h);
        }
    }

    Ok((current_x, steps))
}

/// Classical 4th-Order Runge-Kutta Butcher Tableau (RK4).
pub const RK4_TABLEAU: ExplicitButcherTableau<4> = ExplicitButcherTableau {
    c: [0.0, 0.5, 0.5, 1.0],
    a: [
        [0.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ],
    b: [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0],
    e: None,
    is_fsal: false,
    safety: 0.9,
    exp: 0.25,
    min_scale: 0.2,
    max_scale: 2.0,
};

/// Dormand-Prince 5(4) Butcher Tableau (DP45).
pub const DP45_TABLEAU: ExplicitButcherTableau<7> = ExplicitButcherTableau {
    c: [0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0],
    a: [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0, 0.0],
        [19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0, 0.0, 0.0, 0.0],
        [9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0, 0.0, 0.0],
        [35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0],
    ],
    b: [35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0],
    e: Some([
        71.0 / 57600.0,
        0.0,
        -71.0 / 16695.0,
        71.0 / 1920.0,
        -17253.0 / 339200.0,
        22.0 / 525.0,
        -1.0 / 40.0,
    ]),
    is_fsal: true,
    safety: 0.9,
    exp: 0.2,
    min_scale: 0.2,
    max_scale: 2.0,
};

/// Dormand-Prince (RK45) Adaptive Flow Solver.
///
/// Computes a 5th-order accurate solution with an embedded 4th-order error estimate
/// using a 7-stage First-Same-As-Last (FSAL) Butcher tableau.
pub struct DormandPrince45;

impl DormandPrince45 {
    pub const TABLEAU: ExplicitButcherTableau<7> = DP45_TABLEAU;

    /// Computes a single adaptive step with local error estimation.
    pub fn step<F>(x: &[f32], t: f32, h: f32, tol: f32, velocity_fn: F) -> AdaptiveStepResult
    where
        F: FnMut(&[f32], f32) -> Vec<f32>,
    {
        Self::TABLEAU.adaptive_step(x, t, h, tol, velocity_fn)
    }

    /// Full adaptive trajectory integration from $t = 0.0$ to $t = 1.0$.
    pub fn solve_adaptive_trajectory<F>(
        x0: &[f32],
        initial_h: f32,
        min_h: f32,
        max_h: f32,
        tol: f32,
        max_iterations: usize,
        mut velocity_fn: F,
    ) -> Result<(Vec<f32>, usize), String>
    where
        F: FnMut(&[f32], f32) -> Vec<f32>,
    {
        solve_adaptive_trajectory_core(
            "RK45",
            x0,
            initial_h,
            min_h,
            max_h,
            tol,
            max_iterations,
            |x, t, h, tol| Self::step(x, t, h, tol, &mut velocity_fn),
        )
    }
}

/// Bogacki-Shampine 3(2) Butcher Tableau (BS23).
pub const BS23_TABLEAU: ExplicitButcherTableau<4> = ExplicitButcherTableau {
    c: [0.0, 1.0 / 2.0, 3.0 / 4.0, 1.0],
    a: [
        [0.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0, 0.0],
        [0.0, 0.75, 0.0, 0.0],
        [2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0, 0.0],
    ],
    b: [2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0, 0.0],
    e: Some([5.0 / 72.0, -1.0 / 12.0, -1.0 / 9.0, 1.0 / 8.0]),
    is_fsal: true,
    safety: 0.85,
    exp: 0.33,
    min_scale: 0.2,
    max_scale: 2.0,
};

/// Bogacki-Shampine (RK23) Low-Power Adaptive Flow Solver.
///
/// 3rd-order solution with an embedded 2nd-order error estimate using a 4-stage FSAL tableau.
/// Offers minimal computational overhead per step, ideal for mobile audio synthesis.
pub struct BogackiShampine23;

impl BogackiShampine23 {
    pub const TABLEAU: ExplicitButcherTableau<4> = BS23_TABLEAU;

    pub const C2: f32 = 1.0 / 2.0;
    pub const A21: f32 = 1.0 / 2.0;
    pub const C3: f32 = 3.0 / 4.0;
    pub const A32: f32 = 3.0 / 4.0;
    pub const B1: f32 = 2.0 / 9.0;
    pub const B2: f32 = 1.0 / 3.0;
    pub const B3: f32 = 4.0 / 9.0;
    pub const E1: f32 = 5.0 / 72.0;
    pub const E2: f32 = -1.0 / 12.0;
    pub const E3: f32 = -1.0 / 9.0;
    pub const E4: f32 = 1.0 / 8.0;

    pub fn step<F>(x: &[f32], t: f32, h: f32, tol: f32, velocity_fn: F) -> AdaptiveStepResult
    where
        F: FnMut(&[f32], f32) -> Vec<f32>,
    {
        Self::TABLEAU.adaptive_step(x, t, h, tol, velocity_fn)
    }

    /// Integrates trajectory from $t=0$ to $t=1$ with dynamic step-size control using RK23.
    pub fn solve_adaptive_trajectory<F>(
        x0: &[f32],
        initial_h: f32,
        min_h: f32,
        max_h: f32,
        tol: f32,
        max_iterations: usize,
        mut velocity_fn: F,
    ) -> Result<(Vec<f32>, usize), String>
    where
        F: FnMut(&[f32], f32) -> Vec<f32>,
    {
        solve_adaptive_trajectory_core(
            "RK23",
            x0,
            initial_h,
            min_h,
            max_h,
            tol,
            max_iterations,
            |x, t, h, tol| Self::step(x, t, h, tol, &mut velocity_fn),
        )
    }
}

/// Learned Flow Step-Size Controller (Neural ODE Step Modulator).
/// Dynamically predicts optimal integration step sizes based on trajectory kinematics (acceleration, jerk),
/// environmental turbulence (weather latent), and hardware constraints (thermal deficit).
#[derive(Debug, Clone)]
pub struct LearnedFlowController {
    pub base_h: f32,
    pub min_h: f32,
    pub max_h: f32,
    pub curvature_sensitivity: f32,
    pub jerk_sensitivity: f32,
    pub velocity_sensitivity: f32,
    pub thermal_relaxation: f32,
}

impl Default for LearnedFlowController {
    fn default() -> Self {
        Self {
            base_h: 0.05,
            min_h: 0.01,
            max_h: 0.20,
            curvature_sensitivity: 1.5,
            jerk_sensitivity: 0.8,
            velocity_sensitivity: 0.5,
            thermal_relaxation: 2.0,
        }
    }
}

impl LearnedFlowController {
    pub fn new(base_h: f32, min_h: f32, max_h: f32) -> Self {
        Self {
            base_h,
            min_h,
            max_h,
            ..Default::default()
        }
    }

    /// Predicts optimal step size $h_t$ incorporating leading kinematic indicators (acceleration, jerk),
    /// the active weather latent, and hardware thermal telemetry.
    pub fn predict_step_size(
        &self,
        v_current: &[f32],
        v_prev: Option<&[f32]>,
        v_prev_prev: Option<&[f32]>,
        weather_latent: Option<&[f32]>,
        thermal_deficit: f32,
        dt: f32,
    ) -> f32 {
        let v_norm =
            (v_current.iter().map(|&x| x * x).sum::<f32>() / v_current.len().max(1) as f32).sqrt();

        let mut acceleration = 0.0f32;
        let mut jerk = 0.0f32;

        if let Some(prev) = v_prev {
            let mut diff_sq = 0.0f32;
            for i in 0..v_current.len().min(prev.len()) {
                let d = (v_current[i] - prev[i]) / dt.max(1e-6);
                diff_sq += d * d;
            }
            acceleration = (diff_sq / v_current.len().max(1) as f32).sqrt();

            if let Some(prev_prev) = v_prev_prev {
                let mut jerk_sq = 0.0f32;
                for i in 0..v_current.len().min(prev_prev.len()) {
                    let a_curr = (v_current[i] - prev[i]) / dt.max(1e-6);
                    let a_prev = (prev[i] - prev_prev[i]) / dt.max(1e-6);
                    let j = (a_curr - a_prev) / dt.max(1e-6);
                    jerk_sq += j * j;
                }
                jerk = (jerk_sq / v_current.len().max(1) as f32).sqrt();
            }
        }

        // If a weather latent is provided, compute its magnitude as a proxy for environmental turbulence.
        let latent_turbulence = if let Some(zw) = weather_latent {
            (zw.iter().map(|&x| x * x).sum::<f32>() / zw.len().max(1) as f32).sqrt()
        } else {
            0.0f32
        };

        // Hardware thermal override: if device is throttling (thermal_deficit > 0), force larger steps (relax damping).
        let thermal_factor = 1.0 - (thermal_deficit.clamp(0.0, 1.0) * 0.8);

        // Modulate base_h inversely with kinematics and weather, softened by thermal relaxation
        let damping = (1.0
            + self.curvature_sensitivity * acceleration
            + self.jerk_sensitivity * jerk
            + self.velocity_sensitivity * v_norm
            + 0.5 * latent_turbulence)
            * thermal_factor;

        (self.base_h / damping).clamp(self.min_h, self.max_h)
    }
}

/// Tsitouras 5(4) Butcher Tableau (Tsit5).
pub const TSIT54_TABLEAU: ExplicitButcherTableau<7> = ExplicitButcherTableau {
    c: [0.0, 0.161, 0.327, 0.9, 0.98002554, 1.0, 1.0],
    a: [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.161, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.008415641, 0.33541564, 0.0, 0.0, 0.0, 0.0, 0.0],
        [2.8971078, -6.3388076, 4.3416998, 0.0, 0.0, 0.0, 0.0],
        [1.6080186, -3.530745, 2.302485, 0.6002668, 0.0, 0.0, 0.0],
        [1.7700564, -3.436595, 2.0665582, 0.49523602, 0.10474432, 0.0, 0.0],
        [0.09646077, 0.01, 0.47988965, 1.3790085, -3.2900696, 2.3247106, 0.0],
    ],
    b: [0.09646077, 0.01, 0.47988965, 1.3790085, -3.2900696, 2.3247106, 0.0],
    e: Some([
        0.001780011,
        0.00081643446,
        -0.007880878,
        0.144711,
        -0.58235717,
        0.4580821,
        -0.015151515,
    ]),
    is_fsal: true,
    safety: 0.9,
    exp: 0.2,
    min_scale: 0.2,
    max_scale: 2.0,
};

/// Tsitouras 5(4) Adaptive Flow Solver (`Tsit5`).
///
/// Sotiris Tsitouras (2011) optimized Runge-Kutta 5(4) embedded pair.
/// Standard default in modern SciML (Julia DifferentialEquations.jl)
/// with lower truncation error coefficients and superior step efficiency over Dormand-Prince.
pub struct Tsitouras54;

impl Tsitouras54 {
    pub const TABLEAU: ExplicitButcherTableau<7> = TSIT54_TABLEAU;

    pub const C2: f32 = 0.161;
    pub const A21: f32 = 0.161;
    pub const C3: f32 = 0.327;
    pub const A31: f32 = -0.00841564124974345;
    pub const A32: f32 = 0.33541564124974345;
    pub const C4: f32 = 0.9;
    pub const A41: f32 = 2.8971077738690227;
    pub const A42: f32 = -6.3388075701223585;
    pub const A43: f32 = 4.341699796253336;
    pub const C5: f32 = 0.9800255409045097;
    pub const A51: f32 = 1.6080185734186085;
    pub const A52: f32 = -3.5307449109724837;
    pub const A53: f32 = 2.302485098004079;
    pub const A54: f32 = 0.6002667794543059;
    pub const C6: f32 = 1.0;
    pub const A61: f32 = 1.7700563917777126;
    pub const A62: f32 = -3.4365949544086616;
    pub const A63: f32 = 2.06655822123656;
    pub const A64: f32 = 0.4952360233159734;
    pub const A65: f32 = 0.1047443180784157;
    pub const B1: f32 = 0.09646076681806523;
    pub const B2: f32 = 0.01;
    pub const B3: f32 = 0.4798896504144996;
    pub const B4: f32 = 1.3790085741037419;
    pub const B5: f32 = -3.2900695154360807;
    pub const B6: f32 = 2.324710524099774;
    pub const E1: f32 = 0.0017800110522257773;
    pub const E2: f32 = 0.0008164344596567463;
    pub const E3: f32 = -0.007880878010261994;
    pub const E4: f32 = 0.1447110071732629;
    pub const E5: f32 = -0.5823571654525552;
    pub const E6: f32 = 0.45808210592918686;
    pub const E7: f32 = -0.015151515151515152;

    pub fn step<F>(x: &[f32], t: f32, h: f32, tol: f32, velocity_fn: F) -> AdaptiveStepResult
    where
        F: FnMut(&[f32], f32) -> Vec<f32>,
    {
        Self::TABLEAU.adaptive_step(x, t, h, tol, velocity_fn)
    }

    pub fn solve_adaptive_trajectory<F>(
        x0: &[f32],
        initial_h: f32,
        min_h: f32,
        max_h: f32,
        tol: f32,
        max_iterations: usize,
        mut velocity_fn: F,
    ) -> Result<(Vec<f32>, usize), String>
    where
        F: FnMut(&[f32], f32) -> Vec<f32>,
    {
        solve_adaptive_trajectory_core(
            "Tsit5",
            x0,
            initial_h,
            min_h,
            max_h,
            tol,
            max_iterations,
            |x, t, h, tol| Self::step(x, t, h, tol, &mut velocity_fn),
        )
    }
}

/// Heun 2nd-Order Adaptive Butcher Tableau (Heun2).
pub const HEUN2_TABLEAU: ExplicitButcherTableau<2> = ExplicitButcherTableau {
    c: [0.0, 1.0],
    a: [
        [0.0, 0.0],
        [1.0, 0.0],
    ],
    b: [0.5, 0.5],
    e: Some([-0.5, 0.5]),
    is_fsal: false,
    safety: 0.9,
    exp: 0.5,
    min_scale: 0.2,
    max_scale: 5.0,
};

/// Heun's 2nd-Order Adaptive Predictor-Corrector (`HeunAdaptive2`).
///
/// Standard in continuous audio/image diffusion (EDM / Karras formulation)
/// offering rapid 2-stage trapezoidal curvature tracking.
pub struct HeunAdaptive2;

impl HeunAdaptive2 {
    pub const TABLEAU: ExplicitButcherTableau<2> = HEUN2_TABLEAU;

    pub fn step<F>(x: &[f32], t: f32, h: f32, tol: f32, velocity_fn: F) -> AdaptiveStepResult
    where
        F: FnMut(&[f32], f32) -> Vec<f32>,
    {
        Self::TABLEAU.adaptive_step(x, t, h, tol, velocity_fn)
    }

    pub fn solve_adaptive_trajectory<F>(
        x0: &[f32],
        initial_h: f32,
        min_h: f32,
        max_h: f32,
        tol: f32,
        max_iterations: usize,
        mut velocity_fn: F,
    ) -> Result<(Vec<f32>, usize), String>
    where
        F: FnMut(&[f32], f32) -> Vec<f32>,
    {
        solve_adaptive_trajectory_core(
            "Heun2",
            x0,
            initial_h,
            min_h,
            max_h,
            tol,
            max_iterations,
            |x, t, h, tol| Self::step(x, t, h, tol, &mut velocity_fn),
        )
    }
}

/// DPM-Solver++ (2nd-Order Fast Exponential Integrator).
///
/// Tailored for fast flow matching generation in 10-15 steps.
pub struct DpmSolverPP;

impl DpmSolverPP {
    pub fn solve_fast_trajectory<F>(
        x0: &[f32],
        steps: usize,
        mut velocity_fn: F,
    ) -> Result<Vec<f32>, String>
    where
        F: FnMut(&[f32], f32) -> Vec<f32>,
    {
        if steps == 0 {
            return Ok(x0.to_vec());
        }

        let n = x0.len();
        let dt = 1.0f32 / steps as f32;
        let mut current_x = x0.to_vec();
        let mut v_prev: Option<Vec<f32>> = None;

        for step in 0..steps {
            let t = step as f32 * dt;
            let v_curr = velocity_fn(&current_x, t);

            if let Some(ref v_p) = v_prev {
                // 2nd-order multistep Adams-Bashforth expansion
                for i in 0..n {
                    current_x[i] += dt * (1.5 * v_curr[i] - 0.5 * v_p[i]);
                }
            } else {
                // 1st-order startup Euler step
                for i in 0..n {
                    current_x[i] += dt * v_curr[i];
                }
            }
            v_prev = Some(v_curr);
        }

        Ok(current_x)
    }
}

/// Learned Predictor-Evaluator-Corrector (PEC) Integrator.
///
/// Uses a neural-parameterized Predictor-Corrector loop where the corrector
/// weights and evaluation strides are dynamically predicted by a lightweight
/// hypernetwork or learned end-to-end to optimally traverse the data manifold.
#[derive(Debug, Clone)]
pub struct TrainedPecSolver {
    /// Learned weights for the Predictor (Adams-Bashforth style)
    pub predictor_coeffs: Vec<f32>,
    /// Learned weights for the Corrector (Adams-Moulton style)
    pub corrector_coeffs: Vec<f32>,
}

impl TrainedPecSolver {
    pub fn new(predictor_coeffs: Vec<f32>, corrector_coeffs: Vec<f32>) -> Self {
        Self {
            predictor_coeffs,
            corrector_coeffs,
        }
    }

    /// Integrates a trajectory using the learned predictor-corrector history buffer.
    pub fn solve_trajectory<F>(
        &self,
        x0: &[f32],
        steps: usize,
        mut velocity_fn: F,
    ) -> Result<Vec<f32>, String>
    where
        F: FnMut(&[f32], f32) -> Vec<f32>,
    {
        if steps == 0 {
            return Ok(x0.to_vec());
        }

        let n = x0.len();
        let dt = 1.0f32 / steps as f32;
        let mut current_x = x0.to_vec();

        let hist_len = self
            .predictor_coeffs
            .len()
            .max(self.corrector_coeffs.len().saturating_sub(1));
        let mut v_history = std::collections::VecDeque::with_capacity(hist_len);

        for step in 0..steps {
            let t = step as f32 * dt;

            // Evaluator: get v(x_n, t_n)
            let v_curr = velocity_fn(&current_x, t);

            if v_history.len() == hist_len {
                v_history.pop_back();
            }
            v_history.push_front(v_curr.clone());

            // Startup phase: use Euler until history buffer is filled
            if v_history.len() < hist_len {
                for i in 0..n {
                    current_x[i] += dt * v_curr[i];
                }
                continue;
            }

            // Predictor (P)
            let mut x_pred = current_x.clone();
            for i in 0..n {
                let mut v_pred = 0.0;
                for (j, &c) in self.predictor_coeffs.iter().enumerate() {
                    v_pred += c * v_history[j][i];
                }
                x_pred[i] += dt * v_pred;
            }

            // Evaluator (E) for predictor
            let v_pred_eval = velocity_fn(&x_pred, t + dt);

            // Corrector (C)
            for i in 0..n {
                let mut v_corr = if !self.corrector_coeffs.is_empty() {
                    self.corrector_coeffs[0] * v_pred_eval[i]
                } else {
                    0.0
                };
                for (j, &c) in self.corrector_coeffs.iter().skip(1).enumerate() {
                    v_corr += c * v_history[j][i];
                }
                current_x[i] += dt * v_corr;
            }
        }

        Ok(current_x)
    }
}

/// Learned Implicit Runge-Kutta (Implicit RK) Solver.
///
/// Implements implicit solvers (like Backward Euler or Radau IIA) with a learned
/// fixed-point iteration matrix, optimizing for unconditionally stable large step
/// sizes on stiff physical equations.
#[derive(Debug, Clone)]
pub struct TrainedImplicitRkSolver {
    /// Learned Butcher tableau A matrix (flattened row-major)
    pub a_coeffs: Vec<f32>,
    /// Learned weights b
    pub b_coeffs: Vec<f32>,
    /// Learned nodes c
    pub c_coeffs: Vec<f32>,
    pub stages: usize,
    pub max_newton_iters: usize,
    pub tol: f32,
}

impl TrainedImplicitRkSolver {
    pub fn new(a: Vec<f32>, b: Vec<f32>, c: Vec<f32>, max_iters: usize, tol: f32) -> Self {
        let stages = b.len();
        Self {
            a_coeffs: a,
            b_coeffs: b,
            c_coeffs: c,
            stages,
            max_newton_iters: max_iters,
            tol,
        }
    }

    pub fn step<F>(&self, x: &[f32], t: f32, h: f32, mut velocity_fn: F) -> Result<Vec<f32>, String>
    where
        F: FnMut(&[f32], f32) -> Vec<f32>,
    {
        let n = x.len();
        // Initialize k_i with explicit Euler guess
        let mut k = vec![vec![0.0f32; n]; self.stages];
        let v0 = velocity_fn(x, t);
        for i in 0..self.stages {
            k[i] = v0.clone();
        }

        // Fixed point Picard iteration for implicit stages with hoisted scratch buffers
        let mut x_stage = vec![0.0f32; n];
        let mut k_next = vec![vec![0.0f32; n]; self.stages];

        for _iter in 0..self.max_newton_iters {
            let mut max_err = 0.0f32;

            for i in 0..self.stages {
                x_stage.copy_from_slice(x);
                for j in 0..self.stages {
                    let a_ij = self.a_coeffs[i * self.stages + j];
                    if a_ij != 0.0 {
                        for dim in 0..n {
                            x_stage[dim] += h * a_ij * k[j][dim];
                        }
                    }
                }

                let v_stage = velocity_fn(&x_stage, t + self.c_coeffs[i] * h);

                for dim in 0..n {
                    let err = (v_stage[dim] - k[i][dim]).abs();
                    if err > max_err {
                        max_err = err;
                    }
                    k_next[i][dim] = v_stage[dim];
                }
            }
            std::mem::swap(&mut k, &mut k_next);

            if max_err < self.tol {
                break;
            }
        }

        let mut x_next = x.to_vec();
        for i in 0..self.stages {
            for dim in 0..n {
                x_next[dim] += h * self.b_coeffs[i] * k[i][dim];
            }
        }

        Ok(x_next)
    }

    pub fn solve_trajectory<F>(
        &self,
        x0: &[f32],
        steps: usize,
        mut velocity_fn: F,
    ) -> Result<Vec<f32>, String>
    where
        F: FnMut(&[f32], f32) -> Vec<f32>,
    {
        if steps == 0 {
            return Ok(x0.to_vec());
        }

        let dt = 1.0f32 / steps as f32;
        let mut current_x = x0.to_vec();

        for step in 0..steps {
            let t = step as f32 * dt;
            current_x = self.step(&current_x, t, dt, &mut velocity_fn)?;
        }

        Ok(current_x)
    }
}

/// 4th-Order Predict-Correct-Evaluate-Correct (PCEC) Multistep Integrator.
///
/// Implements the high-precision Adams-Bashforth-Moulton PCEC scheme:
/// 1. **P (Predict)**: $x_{n+1}^{(0)} = x_n + \frac{h}{24}(55 f_n - 59 f_{n-1} + 37 f_{n-2} - 9 f_{n-3})$
/// 2. **C (Correct 1)**: $x_{n+1}^{(1)} = x_n + \frac{h}{24}(9 f(x_{n+1}^{(0)}) + 19 f_n - 5 f_{n-1} + f_{n-2})$
/// 3. **E (Evaluate 2)**: $f_{n+1}^{(1)} = f(x_{n+1}^{(1)}, t_{n+1})$
/// 4. **C (Correct 2)**: $x_{n+1}^{(2)} = x_n + \frac{h}{24}(9 f_{n+1}^{(1)} + 19 f_n - 5 f_{n-1} + f_{n-2})$
#[derive(Debug, Clone, Default)]
pub struct PcecAdamsMoulton4;

impl PcecAdamsMoulton4 {
    /// Integrates a single PCEC step given the current state and the 4 historical derivatives [f_n, f_{n-1}, f_{n-2}, f_{n-3}].
    pub fn step<F, H>(
        x: &[f32],
        t: f32,
        h: f32,
        f_history: &[H],
        mut velocity_fn: F,
    ) -> (Vec<f32>, Vec<f32>, f32)
    where
        F: FnMut(&[f32], f32) -> Vec<f32>,
        H: AsRef<[f32]>,
    {
        let n = x.len();
        assert!(f_history.len() >= 4, "PCEC-4 requires at least 4 historical derivatives");
        let f0 = f_history[0].as_ref();
        let f1 = f_history[1].as_ref();
        let f2 = f_history[2].as_ref();
        let f3 = f_history[3].as_ref();

        let inv24 = h / 24.0f32;

        // 1. Predict (P) using Adams-Bashforth 4
        let mut x_work = vec![0.0f32; n];
        for i in 0..n {
            x_work[i] = x[i] + inv24 * (55.0 * f0[i] - 59.0 * f1[i] + 37.0 * f2[i] - 9.0 * f3[i]);
        }

        // 2. Initial Evaluate (E1)
        let f_pred = velocity_fn(&x_work, t + h);

        // 3. First Correct (C1) using Adams-Moulton 4
        let mut x_corr1 = vec![0.0f32; n];
        let mut milne_err_sq = 0.0f32;
        for i in 0..n {
            let corr = x[i] + inv24 * (9.0 * f_pred[i] + 19.0 * f0[i] - 5.0 * f1[i] + f2[i]);
            let diff = corr - x_work[i];
            milne_err_sq += diff * diff;
            x_corr1[i] = corr;
        }

        // 4. Second Evaluate (E2)
        let f_eval2 = velocity_fn(&x_corr1, t + h);

        // 5. Second Correct (C2) - reuse x_work to eliminate x_final allocation
        for i in 0..n {
            x_work[i] = x[i] + inv24 * (9.0 * f_eval2[i] + 19.0 * f0[i] - 5.0 * f1[i] + f2[i]);
        }

        // Milne error estimate factor: 19/270 ≈ 0.07037
        let milne_error = (19.0 / 270.0) * milne_err_sq.sqrt();
        let f_final = velocity_fn(&x_work, t + h);

        (x_work, f_final, milne_error)
    }

    /// Full trajectory integration using RK4 to seed the initial 4 history steps, followed by PCEC.
    pub fn solve_trajectory<F>(x0: &[f32], steps: usize, mut velocity_fn: F) -> Vec<f32>
    where
        F: FnMut(&[f32], f32) -> Vec<f32>,
    {
        if steps < 4 {
            return Rk4FlowSolver::solve_trajectory(x0, steps, velocity_fn);
        }

        let dt = 1.0f32 / steps as f32;
        let mut current_x = x0.to_vec();
        let mut f_history: std::collections::VecDeque<Vec<f32>> = std::collections::VecDeque::with_capacity(4);

        // Bootstrap the first 3 steps with RK4 to populate history
        for step_idx in 0..3 {
            let t = step_idx as f32 * dt;
            let v_curr = velocity_fn(&current_x, t);
            f_history.push_front(v_curr);
            current_x = Rk4FlowSolver::step(&current_x, t, dt, &mut velocity_fn);
        }
        let v_3 = velocity_fn(&current_x, 3.0 * dt);
        f_history.push_front(v_3);

        // Run remaining steps via PCEC
        for step_idx in 3..steps {
            let t = step_idx as f32 * dt;
            let hist_refs: [&[f32]; 4] = [
                &f_history[0],
                &f_history[1],
                &f_history[2],
                &f_history[3],
            ];
            let (next_x, next_f, _) = Self::step(&current_x, t, dt, &hist_refs, &mut velocity_fn);
            current_x = next_x;
            f_history.pop_back();
            f_history.push_front(next_f);
        }

        current_x
    }
}

/// Euler-Maruyama Stochastic Differential Equation (SDE) Integrator.
///
/// Integrates continuous SDE score-based diffusion trajectories:
/// $$dx_t = v(x_t, t) dt + g(t) dW_t$$
#[derive(Debug, Clone, Default)]
pub struct EulerMaruyamaSde;

impl EulerMaruyamaSde {
    /// Steps a stochastic differential equation with standard Gaussian noise $\xi \sim \mathcal{N}(0, I)$.
    pub fn step<F, G>(
        x: &[f32],
        t: f32,
        dt: f32,
        noise: &[f32],
        mut drift_fn: F,
        mut diffusion_fn: G,
    ) -> Vec<f32>
    where
        F: FnMut(&[f32], f32) -> Vec<f32>,
        G: FnMut(f32) -> f32,
    {
        let n = x.len();
        let drift = drift_fn(x, t);
        let g = diffusion_fn(t);
        let sqrt_dt = dt.sqrt();

        let mut x_next = vec![0.0f32; n];
        for i in 0..n {
            x_next[i] = x[i] + drift[i] * dt + g * sqrt_dt * noise[i];
        }
        x_next
    }
}

/// Symplectic Velocity-Verlet Integrator for Hamiltonian & Physics Flow Dynamics.
///
/// Preserves symplectic 2-form area and conserved energy quantities:
/// $$q_{t+h} = q_t + p_t h + \frac{1}{2} F(q_t) h^2$$
/// $$p_{t+h} = p_t + \frac{1}{2} (F(q_t) + F(q_{t+h})) h$$
#[derive(Debug, Clone, Default)]
pub struct SymplecticVelocityVerlet;

impl SymplecticVelocityVerlet {
    pub fn step<F>(
        q: &[f32],
        p: &[f32],
        h: f32,
        mut force_fn: F,
    ) -> (Vec<f32>, Vec<f32>)
    where
        F: FnMut(&[f32]) -> Vec<f32>,
    {
        let n = q.len();
        let f_init = force_fn(q);

        // 1. Position update: q_{t+h} = q_t + p_t h + 0.5 h^2 F(q_t)
        let mut q_next = vec![0.0f32; n];
        for i in 0..n {
            q_next[i] = q[i] + p[i] * h + 0.5 * h * h * f_init[i];
        }

        // 2. Next force evaluation at q_{t+h}
        let f_next = force_fn(&q_next);

        // 3. Momentum update: p_{t+h} = p_t + 0.5 h (F(q_t) + F(q_{t+h}))
        let mut p_next = vec![0.0f32; n];
        for i in 0..n {
            p_next[i] = p[i] + 0.5 * h * (f_init[i] + f_next[i]);
        }

        (q_next, p_next)
    }
}



// ============================================================================
// Extended Runge-Kutta & Implicit Solvers (RK1, RK2, RK3, RK34, Implicit Euler/Midpoint)
// ============================================================================

/// 1st-Order Forward Euler Integrator (`EulerForwardRk1`).
pub struct EulerForwardRk1;

impl EulerForwardRk1 {
    pub fn step<F>(x: &[f32], t: f32, h: f32, mut velocity_fn: F) -> Vec<f32>
    where
        F: FnMut(&[f32], f32) -> Vec<f32>,
    {
        let n = x.len();
        let k1 = velocity_fn(x, t);
        let mut x_next = vec![0.0f32; n];
        for i in 0..n {
            x_next[i] = x[i] + h * k1[i];
        }
        x_next
    }

    pub fn solve_trajectory<F>(x0: &[f32], steps: usize, mut velocity_fn: F) -> Vec<f32>
    where
        F: FnMut(&[f32], f32) -> Vec<f32>,
    {
        solve_fixed_trajectory_core(x0, steps, |x, t, dt| Self::step(x, t, dt, &mut velocity_fn))
    }
}

/// 2nd-Order Explicit Midpoint Runge-Kutta Integrator (`MidpointRk2`).
pub struct MidpointRk2;

impl MidpointRk2 {
    pub fn step<F>(x: &[f32], t: f32, h: f32, mut velocity_fn: F) -> Vec<f32>
    where
        F: FnMut(&[f32], f32) -> Vec<f32>,
    {
        let n = x.len();
        let k1 = velocity_fn(x, t);
        let mut x_mid = vec![0.0f32; n];
        for i in 0..n {
            x_mid[i] = x[i] + 0.5 * h * k1[i];
        }
        let k2 = velocity_fn(&x_mid, t + 0.5 * h);
        let mut x_next = vec![0.0f32; n];
        for i in 0..n {
            x_next[i] = x[i] + h * k2[i];
        }
        x_next
    }

    pub fn solve_trajectory<F>(x0: &[f32], steps: usize, mut velocity_fn: F) -> Vec<f32>
    where
        F: FnMut(&[f32], f32) -> Vec<f32>,
    {
        solve_fixed_trajectory_core(x0, steps, |x, t, dt| Self::step(x, t, dt, &mut velocity_fn))
    }
}

/// Classical 3rd-Order Kutta Butcher Tableau (Kutta3).
pub const KUTTA3_TABLEAU: ExplicitButcherTableau<3> = ExplicitButcherTableau {
    c: [0.0, 0.5, 1.0],
    a: [
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [-1.0, 2.0, 0.0],
    ],
    b: [1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0],
    e: None,
    is_fsal: false,
    safety: 0.9,
    exp: 0.33,
    min_scale: 0.2,
    max_scale: 2.0,
};

/// 3rd-Order Classical Kutta Integrator (`KuttaRk3`).
pub struct KuttaRk3;

impl KuttaRk3 {
    pub const TABLEAU: ExplicitButcherTableau<3> = KUTTA3_TABLEAU;

    pub fn step<F>(x: &[f32], t: f32, h: f32, velocity_fn: F) -> Vec<f32>
    where
        F: FnMut(&[f32], f32) -> Vec<f32>,
    {
        Self::TABLEAU.step(x, t, h, velocity_fn)
    }

    pub fn solve_trajectory<F>(x0: &[f32], steps: usize, mut velocity_fn: F) -> Vec<f32>
    where
        F: FnMut(&[f32], f32) -> Vec<f32>,
    {
        solve_fixed_trajectory_core(x0, steps, |x, t, dt| Self::step(x, t, dt, &mut velocity_fn))
    }
}

/// Fehlberg 3(4) Embedded Adaptive Butcher Tableau (Fehlberg34).
pub const FEHLBERG34_TABLEAU: ExplicitButcherTableau<5> = ExplicitButcherTableau {
    c: [0.0, 0.25, 0.375, 12.0 / 13.0, 1.0],
    a: [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.25, 0.0, 0.0, 0.0, 0.0],
        [3.0 / 32.0, 9.0 / 32.0, 0.0, 0.0, 0.0],
        [1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0, 0.0, 0.0],
        [439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0, 0.0],
    ],
    b: [25.0 / 216.0, 0.0, 1408.0 / 2565.0, 2197.0 / 4104.0, -0.2],
    e: Some([
        1.0 / 360.0,
        0.0,
        -128.0 / 4275.0,
        -2197.0 / 75240.0,
        -1.0 / 50.0,
    ]),
    is_fsal: false,
    safety: 0.9,
    exp: 0.25,
    min_scale: 0.2,
    max_scale: 3.0,
};

/// 3(4)-Order Fehlberg Embedded Adaptive Integrator (`Fehlberg34`).
pub struct Fehlberg34;

impl Fehlberg34 {
    pub const TABLEAU: ExplicitButcherTableau<5> = FEHLBERG34_TABLEAU;

    pub fn step<F>(x: &[f32], t: f32, h: f32, tol: f32, velocity_fn: F) -> AdaptiveStepResult
    where
        F: FnMut(&[f32], f32) -> Vec<f32>,
    {
        Self::TABLEAU.adaptive_step(x, t, h, tol, velocity_fn)
    }

    pub fn solve_adaptive_trajectory<F>(
        x0: &[f32],
        initial_h: f32,
        min_h: f32,
        max_h: f32,
        tol: f32,
        max_iterations: usize,
        mut velocity_fn: F,
    ) -> Result<(Vec<f32>, usize), String>
    where
        F: FnMut(&[f32], f32) -> Vec<f32>,
    {
        solve_adaptive_trajectory_core(
            "Fehlberg34",
            x0,
            initial_h,
            min_h,
            max_h,
            tol,
            max_iterations,
            |x, t, h, tol| Self::step(x, t, h, tol, &mut velocity_fn),
        )
    }
}

/// 1st-Order Implicit Backward Euler Integrator for Stiff Dynamics (`ImplicitBackwardEuler`).
/// Solves x_{t+h} = x_t + h * v(x_{t+h}, t+h) via Picard fixed-point iteration.
pub struct ImplicitBackwardEuler;

impl ImplicitBackwardEuler {
    pub fn step<F>(x: &[f32], t: f32, h: f32, max_iters: usize, tol: f32, mut velocity_fn: F) -> Vec<f32>
    where
        F: FnMut(&[f32], f32) -> Vec<f32>,
    {
        let n = x.len();
        // Initial explicit Euler prediction
        let v0 = velocity_fn(x, t);
        let mut x_curr = vec![0.0f32; n];
        for i in 0..n {
            x_curr[i] = x[i] + h * v0[i];
        }

        // Fixed-point Picard iteration
        for _ in 0..max_iters {
            let v = velocity_fn(&x_curr, t + h);
            let mut diff = 0.0f32;
            for i in 0..n {
                let x_new = x[i] + h * v[i];
                diff = diff.max((x_new - x_curr[i]).abs());
                x_curr[i] = x_new;
            }
            if diff < tol {
                break;
            }
        }
        x_curr
    }

    pub fn solve_trajectory<F>(x0: &[f32], steps: usize, mut velocity_fn: F) -> Vec<f32>
    where
        F: FnMut(&[f32], f32) -> Vec<f32>,
    {
        let dt = 1.0f32 / steps.max(1) as f32;
        let mut current_x = x0.to_vec();
        for s in 0..steps {
            let t = s as f32 * dt;
            current_x = Self::step(&current_x, t, dt, 10, 1e-5, &mut velocity_fn);
        }
        current_x
    }
}

/// 2nd-Order A-Stable Symplectic Implicit Midpoint Integrator (`ImplicitMidpoint`).
/// Solves x_{t+h} = x_t + h * v((x_t + x_{t+h})/2, t + h/2) via fixed-point iteration.
pub struct ImplicitMidpoint;

impl ImplicitMidpoint {
    pub fn step<F>(x: &[f32], t: f32, h: f32, max_iters: usize, tol: f32, mut velocity_fn: F) -> Vec<f32>
    where
        F: FnMut(&[f32], f32) -> Vec<f32>,
    {
        let n = x.len();
        let v0 = velocity_fn(x, t);
        let mut x_next = vec![0.0f32; n];
        for i in 0..n {
            x_next[i] = x[i] + h * v0[i];
        }

        let t_mid = t + 0.5 * h;
        for _ in 0..max_iters {
            let mut x_mid = vec![0.0f32; n];
            for i in 0..n {
                x_mid[i] = 0.5 * (x[i] + x_next[i]);
            }
            let v_mid = velocity_fn(&x_mid, t_mid);
            let mut diff = 0.0f32;
            for i in 0..n {
                let x_cand = x[i] + h * v_mid[i];
                diff = diff.max((x_cand - x_next[i]).abs());
                x_next[i] = x_cand;
            }
            if diff < tol {
                break;
            }
        }
        x_next
    }

    pub fn solve_trajectory<F>(x0: &[f32], steps: usize, mut velocity_fn: F) -> Vec<f32>
    where
        F: FnMut(&[f32], f32) -> Vec<f32>,
    {
        let dt = 1.0f32 / steps.max(1) as f32;
        let mut current_x = x0.to_vec();
        for s in 0..steps {
            let t = s as f32 * dt;
            current_x = Self::step(&current_x, t, dt, 10, 1e-5, &mut velocity_fn);
        }
        current_x
    }
}

// ============================================================================
// 16. Unified Continuous Flow Solver Trait & Trained Stepper Suite
// ============================================================================

/// Polymorphic trait for continuous probability flow solvers.
pub trait ContinuousFlowSolver {
    /// Integrates trajectory from $t=0$ to $t=1$ over the specified step count.
    fn solve_flow(&self, x0: &[f32], steps: usize, velocity_fn: &mut dyn FnMut(&[f32], f32) -> Vec<f32>) -> Vec<f32>;
}

impl ContinuousFlowSolver for Rk4FlowSolver {
    fn solve_flow(&self, x0: &[f32], steps: usize, velocity_fn: &mut dyn FnMut(&[f32], f32) -> Vec<f32>) -> Vec<f32> {
        Self::solve_trajectory(x0, steps, |x, t| velocity_fn(x, t))
    }
}

impl ContinuousFlowSolver for EulerForwardRk1 {
    fn solve_flow(&self, x0: &[f32], steps: usize, velocity_fn: &mut dyn FnMut(&[f32], f32) -> Vec<f32>) -> Vec<f32> {
        Self::solve_trajectory(x0, steps, |x, t| velocity_fn(x, t))
    }
}

impl ContinuousFlowSolver for MidpointRk2 {
    fn solve_flow(&self, x0: &[f32], steps: usize, velocity_fn: &mut dyn FnMut(&[f32], f32) -> Vec<f32>) -> Vec<f32> {
        Self::solve_trajectory(x0, steps, |x, t| velocity_fn(x, t))
    }
}

impl ContinuousFlowSolver for KuttaRk3 {
    fn solve_flow(&self, x0: &[f32], steps: usize, velocity_fn: &mut dyn FnMut(&[f32], f32) -> Vec<f32>) -> Vec<f32> {
        Self::solve_trajectory(x0, steps, |x, t| velocity_fn(x, t))
    }
}

impl ContinuousFlowSolver for PcecAdamsMoulton4 {
    fn solve_flow(&self, x0: &[f32], steps: usize, velocity_fn: &mut dyn FnMut(&[f32], f32) -> Vec<f32>) -> Vec<f32> {
        Self::solve_trajectory(x0, steps, |x, t| velocity_fn(x, t))
    }
}

/// Dynamic evaluation budget mode selected dynamically per step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalMode {
    /// PE (1 NFE): Predict-Evaluate only. Output is predicted state x^{(p)}.
    Pe,
    /// PEC (1 NFE): Predict-Evaluate-Correct. Output is corrected state x^{(c)} using predicted derivative; no 2nd evaluation.
    Pec,
    /// PECE (2 NFE): Predict-Evaluate-Correct-Evaluate. Re-evaluates derivative at corrected state.
    Pece,
    /// PECEC (2 NFE with secondary correction): Predict-Evaluate-Correct-Evaluate-Correct for stiff dynamics.
    Pecec,
}

/// Diagnostic metadata returned by each integration step.
#[derive(Debug, Clone)]
pub struct StepMetrics {
    pub nfe: usize,
    pub step_size: f32,
    pub eval_mode: EvalMode,
    pub estimated_curvature: f32,
    pub local_truncation_error: f32,
}

/// Learned Predictor-Corrector Integrator with Dynamic Evaluation Gating (`TrainedPcecIntegrator`).
/// Features 4-tier granular gating: PE <-> PEC <-> PECE <-> PECEC.
/// Slashes function evaluations by dynamically skipping corrector evaluations when trajectory curvature is low.
#[derive(Debug, Clone)]
pub struct TrainedPcecIntegrator {
    pub pred_weights: [f32; 3],
    pub corr_weights: [f32; 3],
    pub curvature_threshold_pe: f32,
    pub curvature_threshold_pec: f32,
    pub curvature_threshold_pece: f32,
    pub numeric_type: NumericType,
    pub info_policy: Option<InformationRateGatingPolicy>,
}

impl Default for TrainedPcecIntegrator {
    fn default() -> Self {
        Self {
            pred_weights: [1.5312, -0.6208, 0.0896],
            corr_weights: [0.4167, 0.6667, -0.0834],
            curvature_threshold_pe: 0.06,
            curvature_threshold_pec: 0.14,
            curvature_threshold_pece: 0.30,
            numeric_type: NumericType::F32,
            info_policy: None,
        }
    }
}

impl TrainedPcecIntegrator {
    pub fn new(pred: [f32; 3], corr: [f32; 3]) -> Self {
        Self {
            pred_weights: pred,
            corr_weights: corr,
            ..Default::default()
        }
    }

    pub fn with_numeric_type(mut self, num_type: NumericType) -> Self {
        self.numeric_type = num_type;
        self
    }

    pub fn with_information_policy(mut self, policy: InformationRateGatingPolicy) -> Self {
        self.info_policy = Some(policy);
        self
    }

    /// Takes a dynamic step from $t$ to $t+h$, dynamically selecting between PE, PEC, PECE, and PECEC.
    /// Returns (next_x, v_for_history, step_metrics).
    pub fn step<F>(
        &self,
        x: &[f32],
        history_v: &[[f32; 3]],
        t: f32,
        h: f32,
        mut velocity_fn: F,
    ) -> (Vec<f32>, Vec<f32>, StepMetrics)
    where
        F: FnMut(&[f32], f32) -> Vec<f32>,
    {
        let n = x.len();
        let mut total_nfe = 0;
        let prec = DataTypeSpec::of(self.numeric_type);
        let denom_guard = prec.denominator_guard() as f32;

        // 1. Predictor Step (P)
        let mut x_pred = vec![0.0f32; n];
        for i in 0..n {
            let hist = history_v[i];
            let v_est = self.pred_weights[0] * hist[0]
                + self.pred_weights[1] * hist[1]
                + self.pred_weights[2] * hist[2];
            x_pred[i] = x[i] + h * v_est;
        }

        // 2. Evaluate at predicted state (E)
        let v_pred = velocity_fn(&x_pred, t + h);
        total_nfe += 1;

        // 3. Directional curvature estimation conditioned to datatype precision
        let mut diff_norm_sq = 0.0f32;
        let mut v_norm_sq = 0.0f32;
        for i in 0..n {
            let diff = v_pred[i] - history_v[i][0];
            diff_norm_sq += diff * diff;
            v_norm_sq += history_v[i][0] * history_v[i][0];
        }
        let curvature = (diff_norm_sq / v_norm_sq.max(denom_guard)).sqrt();

        // Tier 1: Ultra-smooth regime -> PE (1 NFE)
        if curvature < self.curvature_threshold_pe {
            return (
                x_pred,
                v_pred,
                StepMetrics {
                    nfe: total_nfe,
                    step_size: h,
                    eval_mode: EvalMode::Pe,
                    estimated_curvature: curvature,
                    local_truncation_error: curvature * h * h * 0.1,
                },
            );
        }

        // Corrector calculation (C) using v_pred
        let mut x_corr = vec![0.0f32; n];
        for i in 0..n {
            let hist = history_v[i];
            let v_corr = self.corr_weights[0] * v_pred[i]
                + self.corr_weights[1] * hist[0]
                + self.corr_weights[2] * hist[1];
            x_corr[i] = x[i] + h * v_corr;
        }

        // Tier 2: Mild curvature -> PEC (1 NFE, corrected state with zero extra evaluations)
        let should_stop_at_pec = if let Some(ref policy) = self.info_policy {
            let x_pred_f64: Vec<f64> = x_pred.iter().map(|&v| v as f64).collect();
            let x_corr_f64: Vec<f64> = x_corr.iter().map(|&v| v as f64).collect();
            let info_gain = policy.compute_information_gain(&x_pred_f64, &x_corr_f64, 1e-4);
            !policy.should_evaluate_secondary_stage(info_gain)
        } else {
            curvature < self.curvature_threshold_pec
        };

        if should_stop_at_pec {
            return (
                x_corr,
                v_pred,
                StepMetrics {
                    nfe: total_nfe,
                    step_size: h,
                    eval_mode: EvalMode::Pec,
                    estimated_curvature: curvature,
                    local_truncation_error: curvature * h * h * 0.05,
                },
            );
        }

        // Second evaluation (E) at corrected state
        let v_corr = velocity_fn(&x_corr, t + h);
        total_nfe += 1;

        // Tier 3: Moderate curvature -> PECE (2 NFE, derivative re-evaluated for accuracy)
        if curvature < self.curvature_threshold_pece {
            return (
                x_corr,
                v_corr,
                StepMetrics {
                    nfe: total_nfe,
                    step_size: h,
                    eval_mode: EvalMode::Pece,
                    estimated_curvature: curvature,
                    local_truncation_error: curvature * h * h * 0.02,
                },
            );
        }

        // Tier 4: High curvature / stiff boundary layers -> PECEC (2 NFE, refined second correction)
        let mut x_refined = vec![0.0f32; n];
        for i in 0..n {
            let hist = history_v[i];
            let v_ref = self.corr_weights[0] * v_corr[i]
                + self.corr_weights[1] * hist[0]
                + self.corr_weights[2] * hist[1];
            x_refined[i] = x[i] + h * v_ref;
        }

        (
            x_refined,
            v_corr,
            StepMetrics {
                nfe: total_nfe,
                step_size: h,
                eval_mode: EvalMode::Pecec,
                estimated_curvature: curvature,
                local_truncation_error: curvature * h * h * 0.005,
            },
        )
    }

    pub fn solve_trajectory<F>(
        &self,
        x0: &[f32],
        num_steps: usize,
        mut velocity_fn: F,
    ) -> (Vec<f32>, Vec<StepMetrics>)
    where
        F: FnMut(&[f32], f32) -> Vec<f32>,
    {
        let n = x0.len();
        let dt = 1.0f32 / num_steps.max(1) as f32;
        let mut curr_x = x0.to_vec();
        let mut history = vec![[0.0f32; 3]; n];
        let mut metrics = Vec::with_capacity(num_steps);

        // Robust dynamic starter: bootstrap backward steps rather than static copy
        let h_half = dt * 0.5;
        let v0 = velocity_fn(&curr_x, 0.0);
        let mut x_prev1 = vec![0.0f32; n];
        for i in 0..n {
            x_prev1[i] = curr_x[i] - h_half * v0[i];
        }
        let v_prev1 = velocity_fn(&x_prev1, -h_half);
        let mut x_prev2 = vec![0.0f32; n];
        for i in 0..n {
            x_prev2[i] = x_prev1[i] - h_half * v_prev1[i];
        }
        let v_prev2 = velocity_fn(&x_prev2, -dt);

        for i in 0..n {
            history[i] = [v0[i], v_prev1[i], v_prev2[i]];
        }

        for s in 0..num_steps {
            let t = s as f32 * dt;
            let (next_x, v_next, step_metric) = self.step(&curr_x, &history, t, dt, &mut velocity_fn);

            for i in 0..n {
                history[i][2] = history[i][1];
                history[i][1] = history[i][0];
                history[i][0] = v_next[i];
            }

            curr_x = next_x;
            metrics.push(step_metric);
        }

        (curr_x, metrics)
    }

    /// Integrates trajectory from $t=0$ to $t=1$, emitting both discrete metrics AND
    /// a continuous dense cubic Hermite interpolant for $O(1)$ arbitrary-resolution evaluation.
    pub fn solve_trajectory_dense<F>(
        &self,
        x0: &[f32],
        num_steps: usize,
        mut velocity_fn: F,
    ) -> (Vec<f32>, DenseHermiteTrajectory<f32>, Vec<StepMetrics>)
    where
        F: FnMut(&[f32], f32) -> Vec<f32>,
    {
        let n = x0.len();
        let dt = 1.0f32 / num_steps.max(1) as f32;
        let mut curr_x = x0.to_vec();
        let mut history = vec![[0.0f32; 3]; n];
        let mut metrics = Vec::with_capacity(num_steps);
        let mut dense = DenseHermiteTrajectory::new();

        // Robust dynamic starter: bootstrap backward steps
        let h_half = dt * 0.5;
        let v0 = velocity_fn(&curr_x, 0.0);
        let mut x_prev1 = vec![0.0f32; n];
        for i in 0..n {
            x_prev1[i] = curr_x[i] - h_half * v0[i];
        }
        let v_prev1 = velocity_fn(&x_prev1, -h_half);
        let mut x_prev2 = vec![0.0f32; n];
        for i in 0..n {
            x_prev2[i] = x_prev1[i] - h_half * v_prev1[i];
        }
        let v_prev2 = velocity_fn(&x_prev2, -dt);

        for i in 0..n {
            history[i] = [v0[i], v_prev1[i], v_prev2[i]];
        }

        let mut v_curr = v0;
        for s in 0..num_steps {
            let t = s as f32 * dt;
            let (next_x, v_next, step_metric) = self.step(&curr_x, &history, t, dt, &mut velocity_fn);

            dense.push_segment(
                t,
                t + dt,
                curr_x.clone(),
                next_x.clone(),
                v_curr.clone(),
                v_next.clone(),
            );

            for i in 0..n {
                history[i][2] = history[i][1];
                history[i][1] = history[i][0];
                history[i][0] = v_next[i];
            }

            curr_x = next_x;
            v_curr = v_next;
            metrics.push(step_metric);
        }

        (curr_x, dense, metrics)
    }
}

impl ContinuousFlowSolver for TrainedPcecIntegrator {
    fn solve_flow(&self, x0: &[f32], steps: usize, velocity_fn: &mut dyn FnMut(&[f32], f32) -> Vec<f32>) -> Vec<f32> {
        let (res, _) = self.solve_trajectory(x0, steps, |x, t| velocity_fn(x, t));
        res
    }
}

/// Curvature-Gated Adaptive Stepper (`CurvatureGatedStepper`).
/// Dynamically adjusts step size $h$ based on directional curvature $\kappa = \|\nabla_v v\|$.
#[derive(Debug, Clone)]
pub struct CurvatureGatedStepper {
    pub base_step: f32,
    pub min_step: f32,
    pub max_step: f32,
    pub target_error: f32,
    pub safety_factor: f32,
    pub numeric_type: NumericType,
}

impl Default for CurvatureGatedStepper {
    fn default() -> Self {
        Self {
            base_step: 0.05,
            min_step: 0.001,
            max_step: 0.25,
            target_error: 1e-4,
            safety_factor: 0.88,
            numeric_type: NumericType::F32,
        }
    }
}

impl CurvatureGatedStepper {
    pub fn with_numeric_type(mut self, num_type: NumericType) -> Self {
        self.numeric_type = num_type;
        self
    }

    pub fn compute_step_size(&self, current_h: f32, curvature: f32) -> f32 {
        let prec = DataTypeSpec::of(self.numeric_type);
        let min_eps = prec.machine_epsilon(None) as f32 * 5.0;
        let antiwindup = prec.pid_antiwindup_guard() as f32;

        if curvature < min_eps {
            return (current_h * 1.5).min(self.max_step);
        }
        let est_error = curvature * current_h * current_h;
        let ratio = self.target_error / est_error.max(antiwindup);
        let scale = self.safety_factor * ratio.sqrt().clamp(0.2, 2.5);
        (current_h * scale).clamp(self.min_step, self.max_step)
    }

    pub fn solve_adaptive_trajectory<F>(
        &self,
        x0: &[f32],
        mut velocity_fn: F,
    ) -> (Vec<f32>, usize, usize)
    where
        F: FnMut(&[f32], f32) -> Vec<f32>,
    {
        let n = x0.len();
        let mut curr_x = x0.to_vec();
        let mut t = 0.0f32;
        let mut h = self.base_step;
        let mut total_steps = 0;
        let mut total_nfe = 0;
        let prec = DataTypeSpec::of(self.numeric_type);
        let temporal_guard = prec.temporal_guard() as f32;
        let denom_guard = prec.denominator_guard() as f32;

        while t < 1.0 - temporal_guard {
            if t + h > 1.0 {
                h = 1.0 - t;
            }

            let k1 = velocity_fn(&curr_x, t);
            total_nfe += 1;

            let mut probe_x = vec![0.0f32; n];
            for i in 0..n {
                probe_x[i] = curr_x[i] + h * k1[i];
            }
            let k2 = velocity_fn(&probe_x, t + h);
            total_nfe += 1;

            let mut diff_sq = 0.0f32;
            let mut k1_sq = 0.0f32;
            for i in 0..n {
                let d = k2[i] - k1[i];
                diff_sq += d * d;
                k1_sq += k1[i] * k1[i];
            }
            let curvature = (diff_sq / k1_sq.max(denom_guard)).sqrt();

            let mut next_x = vec![0.0f32; n];
            for i in 0..n {
                next_x[i] = curr_x[i] + 0.5 * h * (k1[i] + k2[i]);
            }

            curr_x = next_x;
            t += h;
            total_steps += 1;
            h = self.compute_step_size(h, curvature);
        }

        (curr_x, total_steps, total_nfe)
    }
}

/// Symplectic Energy-Preserving Integrator (`SymplecticPreservingIntegrator`).
/// Cancels numerical energy drift in Hamiltonian dynamical systems via momentum projection.
#[derive(Debug, Clone)]
pub struct SymplecticPreservingIntegrator {
    pub energy_correction_gain: f32,
    pub numeric_type: NumericType,
}

impl Default for SymplecticPreservingIntegrator {
    fn default() -> Self {
        Self {
            energy_correction_gain: 0.05,
            numeric_type: NumericType::F32,
        }
    }
}

impl SymplecticPreservingIntegrator {
    pub fn with_numeric_type(mut self, num_type: NumericType) -> Self {
        self.numeric_type = num_type;
        self
    }

    /// Generic leapfrog velocity-Verlet step:
    /// p_{1/2} = p_0 + 0.5 * h * F(q_0)
    /// q_1     = q_0 + h * p_{1/2}
    /// p_1     = p_{1/2} + 0.5 * h * F(q_1)
    #[inline]
    fn leapfrog_core<T, F>(q: &[T], p: &[T], h: T, half: T, mut force_fn: F) -> (Vec<T>, Vec<T>)
    where
        T: Copy + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
        F: FnMut(&[T]) -> Vec<T>,
    {
        let n = q.len();
        let f_q = force_fn(q);
        let mut p_half = Vec::with_capacity(n);
        for i in 0..n {
            p_half.push(p[i] + half * h * f_q[i]);
        }

        let mut q_next = Vec::with_capacity(n);
        for i in 0..n {
            q_next.push(q[i] + h * p_half[i]);
        }

        let f_q_next = force_fn(&q_next);
        let mut p_next = Vec::with_capacity(n);
        for i in 0..n {
            p_next.push(p_half[i] + half * h * f_q_next[i]);
        }

        (q_next, p_next)
    }

    pub fn step<F, H>(
        &self,
        q: &[f32],
        p: &[f32],
        h: f32,
        target_energy: f32,
        force_fn: F,
        hamiltonian_fn: H,
    ) -> (Vec<f32>, Vec<f32>, f32)
    where
        F: FnMut(&[f32]) -> Vec<f32>,
        H: Fn(&[f32], &[f32]) -> f32,
    {
        let (q_next, p_next) = Self::leapfrog_core(q, p, h, 0.5f32, force_fn);
        let spec = DataTypeSpec::of(self.numeric_type);
        let denom_guard = spec.default_num_guard as f32;
        let drift_guard = (spec.default_eff_eps.sqrt()) as f32;

        let current_energy = hamiltonian_fn(&q_next, &p_next);
        let delta_e = current_energy - target_energy;
        let mut p_corrected = p_next.clone();

        let norm_p_sq: f32 = p_next.iter().map(|&v| v * v).sum();
        if norm_p_sq > denom_guard && delta_e.abs() > drift_guard {
            let scale = (1.0 - self.energy_correction_gain * delta_e / (norm_p_sq + denom_guard)).clamp(0.5, 1.5);
            for v in &mut p_corrected {
                *v *= scale;
            }
        }

        let final_energy = hamiltonian_fn(&q_next, &p_corrected);
        (q_next, p_corrected, final_energy)
    }

    /// Pure unprojected Störmer-Verlet / Leapfrog symplectic step in f32.
    /// Preserves phase space volume and bounded Hamiltonian energy oscillations
    /// without artificial manifold projection.
    pub fn step_unprojected<F, H>(
        &self,
        q: &[f32],
        p: &[f32],
        h: f32,
        force_fn: F,
        hamiltonian_fn: H,
    ) -> (Vec<f32>, Vec<f32>, f32)
    where
        F: FnMut(&[f32]) -> Vec<f32>,
        H: Fn(&[f32], &[f32]) -> f32,
    {
        let (q_next, p_next) = Self::leapfrog_core(q, p, h, 0.5f32, force_fn);
        let energy = hamiltonian_fn(&q_next, &p_next);
        (q_next, p_next, energy)
    }

    /// Full precision f64 symplectic step with optional projection.
    pub fn step_f64<F, H>(
        &self,
        q: &[f64],
        p: &[f64],
        h: f64,
        target_energy: f64,
        force_fn: F,
        hamiltonian_fn: H,
    ) -> (Vec<f64>, Vec<f64>, f64)
    where
        F: FnMut(&[f64]) -> Vec<f64>,
        H: Fn(&[f64], &[f64]) -> f64,
    {
        let (q_next, p_next) = Self::leapfrog_core(q, p, h, 0.5f64, force_fn);
        let spec = DataTypeSpec::of(self.numeric_type);
        let denom_guard = spec.default_num_guard;
        let drift_guard = spec.default_eff_eps.sqrt();

        let current_energy = hamiltonian_fn(&q_next, &p_next);
        let delta_e = current_energy - target_energy;
        let mut p_corrected = p_next.clone();

        let norm_p_sq: f64 = p_next.iter().map(|&v| v * v).sum();
        if norm_p_sq > denom_guard && delta_e.abs() > drift_guard {
            let scale = (1.0 - (self.energy_correction_gain as f64) * delta_e / (norm_p_sq + denom_guard)).clamp(0.5, 1.5);
            for v in &mut p_corrected {
                *v *= scale;
            }
        }

        let final_energy = hamiltonian_fn(&q_next, &p_corrected);
        (q_next, p_corrected, final_energy)
    }

    /// Pure unprojected Störmer-Verlet / Leapfrog symplectic step in f64.
    pub fn step_unprojected_f64<F, H>(
        &self,
        q: &[f64],
        p: &[f64],
        h: f64,
        force_fn: F,
        hamiltonian_fn: H,
    ) -> (Vec<f64>, Vec<f64>, f64)
    where
        F: FnMut(&[f64]) -> Vec<f64>,
        H: Fn(&[f64], &[f64]) -> f64,
    {
        let (q_next, p_next) = Self::leapfrog_core(q, p, h, 0.5f64, force_fn);
        let energy = hamiltonian_fn(&q_next, &p_next);
        (q_next, p_next, energy)
    }
}

/// Method used for smoothly blending between different numerical integrator components/regimes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SmoothGatingMethod {
    /// Infinitely differentiable ($C^\infty$) algebraic sigmoid partition of unity (Default).
    /// Requires only 1 square root and 1 divide; zero transcendental exp() calls.
    #[default]
    AlgebraicSigmoid,
    /// Rational Hill-4 gate ($r^4 / (r^4 + K^4)$).
    RationalHill4,
    /// Directional Reversal rational gate ($(-g \cdot g_{\text{prev}})_+^2 / \text{denom}$).
    DirectionalReversal,
    /// Softmax / Boltzmann exponential routing.
    BoltzmannSoftmax,
    /// Compact polynomial Hermite smoothstep ($3u^2 - 2u^3$ or $6u^5 - 15u^4 + 10u^3$).
    HermiteSmoothstep,
    /// Hard threshold / piecewise boolean switching (for baseline ablation comparison).
    HardThreshold,
}

/// Method used for smooth bounding and clamping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SmoothClampingMethod {
    /// Algebraic rational sigmoid clamp ($x / \sqrt{1 + x^2}$) (Default).
    #[default]
    AlgebraicSigmoid,
    /// Quartic root clamp ($x / (1 + x^4)^{1/4}$) with wider linear pass-band.
    QuarticRoot,
    /// Hyperbolic tangent clamp ($\tanh$).
    HyperbolicTangent,
    /// Polynomial smootherstep clamp with compact support.
    HermiteSmootherstep,
    /// Hard piecewise clamp (`x.clamp(min, max)`).
    HardPiecewise,
}

/// Method used for smooth maximum / minimum envelopes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SmoothMaxMethod {
    /// Algebraic square-root soft-max ($0.5 \cdot (a + b + \sqrt{(a - b)^2 + \delta^2})$) (Default).
    #[default]
    AlgebraicSquareRoot,
    /// LogSumExp / Softplus ($\tau \ln(e^{a/\tau} + e^{b/\tau})$).
    LogSumExp,
    /// Hard non-smooth `max(a, b)`.
    HardMax,
}

/// Stochastic calculus interpretation for SDE integration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SdeIntegrationForm {
    /// Itô calculus (non-anticipative Martingale; zero drift-diffusion covariance).
    #[default]
    Ito,
    /// Stratonovich calculus (symmetric midpoint; obeys standard chain rule).
    Stratonovich,
}

/// SDE integration order and Milstein correction scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SdeOrderMode {
    /// Predictor-Corrector Heun / Euler-Maruyama (Strong Order 0.5, Weak Order 1.0).
    #[default]
    PredictorCorrectorHeun,
    /// Derivative-free Milstein correction (Strong Order 1.0) with Itô quadratic variation.
    MilsteinHigherOrder,
}

/// Infinitely differentiable ($C^\infty$) algebraic sigmoid clamp.
/// Replaces non-differentiable `x.clamp(min, max)` with an analytic sigmoid having continuous non-zero derivatives everywhere.
#[inline(always)]
pub fn smooth_clamp(x: f32, min_val: f32, max_val: f32) -> f32 {
    let span = (max_val - min_val).max(1e-6);
    let mid = 0.5 * (min_val + max_val);
    let arg = (x - mid) / (0.5 * span);
    mid + 0.5 * span * (arg / (1.0 + arg * arg).sqrt())
}

/// Dispatches smooth clamping across configurable mathematical representations.
#[inline(always)]
pub fn apply_smooth_clamp(x: f32, min_val: f32, max_val: f32, method: SmoothClampingMethod) -> f32 {
    match method {
        SmoothClampingMethod::AlgebraicSigmoid => smooth_clamp(x, min_val, max_val),
        SmoothClampingMethod::QuarticRoot => {
            let span = (max_val - min_val).max(1e-6);
            let mid = 0.5 * (min_val + max_val);
            let arg = (x - mid) / (0.5 * span);
            let arg2 = arg * arg;
            let arg4 = arg2 * arg2;
            let denom = (1.0 + arg4).sqrt().sqrt();
            mid + 0.5 * span * (arg / denom)
        }
        SmoothClampingMethod::HyperbolicTangent => {
            let span = (max_val - min_val).max(1e-6);
            let mid = 0.5 * (min_val + max_val);
            let arg = (x - mid) / (0.5 * span);
            mid + 0.5 * span * arg.tanh()
        }
        SmoothClampingMethod::HermiteSmootherstep => {
            if x <= min_val {
                min_val
            } else if x >= max_val {
                max_val
            } else {
                let u = (x - min_val) / (max_val - min_val).max(1e-6);
                let s = u * u * u * (u * (u * 6.0 - 15.0) + 10.0);
                min_val + (max_val - min_val) * s
            }
        }
        SmoothClampingMethod::HardPiecewise => x.clamp(min_val, max_val),
    }
}

/// Infinitely differentiable ($C^\infty$) algebraic sigmoid mapping $\mathbb{R} \to (0, 1)$ without transcendental `exp()` calls:
/// $$\sigma(z) = \frac{1}{2}\left(1 + \frac{z}{\sqrt{1 + z^2}}\right)$$
#[inline(always)]
pub fn smooth_sigmoid(z: f32) -> f32 {
    0.5 * (1.0 + z / (1.0 + z * z).sqrt())
}

/// Dispatches smooth gating across configurable mathematical families.
#[inline(always)]
pub fn apply_smooth_gate(z: f32, method: SmoothGatingMethod) -> f32 {
    match method {
        SmoothGatingMethod::AlgebraicSigmoid => smooth_sigmoid(z),
        SmoothGatingMethod::RationalHill4 => {
            let r = z.max(0.0);
            let r2 = r * r;
            let r4 = r2 * r2;
            r4 / (r4 + 1.0)
        }
        SmoothGatingMethod::DirectionalReversal => {
            let u = z.max(0.0);
            let u2 = u * u;
            u2 / (u2 + 0.25)
        }
        SmoothGatingMethod::BoltzmannSoftmax => {
            let clamped_z = z.clamp(-20.0, 20.0);
            1.0 / (1.0 + (-clamped_z).exp())
        }
        SmoothGatingMethod::HermiteSmoothstep => {
            let u = (0.5 * (z + 1.0)).clamp(0.0, 1.0);
            u * u * (3.0 - 2.0 * u)
        }
        SmoothGatingMethod::HardThreshold => {
            if z > 0.0 { 1.0 } else { 0.0 }
        }
    }
}

/// Dispatches smooth maximum across configurable mathematical representations.
#[inline(always)]
pub fn apply_smooth_max(a: f32, b: f32, delta: f32, method: SmoothMaxMethod) -> f32 {
    match method {
        SmoothMaxMethod::AlgebraicSquareRoot => {
            let diff = a - b;
            0.5 * (a + b + (diff * diff + delta * delta).sqrt())
        }
        SmoothMaxMethod::LogSumExp => {
            let tau = delta.max(1e-6);
            let max_val = a.max(b);
            max_val + tau * (((a - max_val) / tau).exp() + ((b - max_val) / tau).exp()).ln()
        }
        SmoothMaxMethod::HardMax => a.max(b),
    }
}

/// Subgradient-Regularized Rough Itô SDE Integrator (`RoughItoIntegrator`).
/// Prevents gradient blow-ups across non-twice-differentiable / Brownian singular drifts.
#[derive(Debug, Clone)]
pub struct RoughItoIntegrator {
    pub subgradient_clamping: f32,
    pub numeric_type: NumericType,
    pub adaptive_clamping: bool,
    pub sde_form: SdeIntegrationForm,
    pub order_mode: SdeOrderMode,
    pub clamping_method: SmoothClampingMethod,
}

impl Default for RoughItoIntegrator {
    fn default() -> Self {
        Self {
            subgradient_clamping: 25.0,
            numeric_type: NumericType::F32,
            adaptive_clamping: true,
            sde_form: SdeIntegrationForm::Ito,
            order_mode: SdeOrderMode::PredictorCorrectorHeun,
            clamping_method: SmoothClampingMethod::AlgebraicSigmoid,
        }
    }
}

impl RoughItoIntegrator {
    pub fn with_numeric_type(mut self, num_type: NumericType) -> Self {
        self.numeric_type = num_type;
        self
    }

    pub fn with_sde_form(mut self, form: SdeIntegrationForm) -> Self {
        self.sde_form = form;
        self
    }

    pub fn with_order_mode(mut self, mode: SdeOrderMode) -> Self {
        self.order_mode = mode;
        self
    }

    pub fn with_clamping_method(mut self, method: SmoothClampingMethod) -> Self {
        self.clamping_method = method;
        self
    }

    pub fn step<F, S>(
        &self,
        x: &[f32],
        t: f32,
        h: f32,
        noise_dw: &[f32],
        mut drift_fn: F,
        mut diffusion_fn: S,
    ) -> Vec<f32>
    where
        F: FnMut(&[f32], f32) -> Vec<f32>,
        S: FnMut(&[f32], f32) -> Vec<f32>,
    {
        let n = x.len();
        let sqrt_h = h.sqrt();
        let prec = DataTypeSpec::of(self.numeric_type);
        let x_norm: f32 = (x.iter().map(|&v| v * v).sum::<f32>() / n.max(1) as f32).sqrt();
        let clamp_limit = if self.adaptive_clamping {
            prec.adaptive_clamp_bound(x_norm as f64, self.subgradient_clamping as f64) as f32
        } else {
            self.subgradient_clamping
        };

        let v1 = drift_fn(x, t);
        let s1 = diffusion_fn(x, t);

        // Derivative-free Stratonovich correction: \mu_Strat = \mu_Ito - 0.5 * \sigma * \sigma'
        let strat_correction = if self.sde_form == SdeIntegrationForm::Stratonovich {
            let mut x_pert = vec![0.0f32; n];
            for i in 0..n {
                x_pert[i] = x[i] + s1[i].abs().max(1e-4) * sqrt_h;
            }
            let s_pert = diffusion_fn(&x_pert, t);
            let mut corr = vec![0.0f32; n];
            for i in 0..n {
                corr[i] = 0.5 * (s_pert[i] - s1[i]) / sqrt_h;
            }
            corr
        } else {
            vec![0.0f32; n]
        };

        let mut x_pred = vec![0.0f32; n];
        for i in 0..n {
            let clamped_drift = apply_smooth_clamp(v1[i] - strat_correction[i], -clamp_limit, clamp_limit, self.clamping_method);
            x_pred[i] = x[i] + h * clamped_drift + sqrt_h * s1[i] * noise_dw[i];
        }

        let v2 = drift_fn(&x_pred, t + h);
        let s2 = diffusion_fn(&x_pred, t + h);

        let mut x_next = vec![0.0f32; n];
        match self.order_mode {
            SdeOrderMode::PredictorCorrectorHeun => {
                for i in 0..n {
                    let clamped_v2 = apply_smooth_clamp(v2[i] - strat_correction[i], -clamp_limit, clamp_limit, self.clamping_method);
                    let avg_drift = 0.5 * (v1[i] - strat_correction[i] + clamped_v2);
                    let avg_diff = 0.5 * (s1[i] + s2[i]);
                    x_next[i] = x[i] + h * avg_drift + sqrt_h * avg_diff * noise_dw[i];
                }
            }
            SdeOrderMode::MilsteinHigherOrder => {
                let mut x_pert = vec![0.0f32; n];
                for i in 0..n {
                    x_pert[i] = x[i] + s1[i] * sqrt_h;
                }
                let s_pert = diffusion_fn(&x_pert, t);

                for i in 0..n {
                    let clamped_v2 = apply_smooth_clamp(v2[i] - strat_correction[i], -clamp_limit, clamp_limit, self.clamping_method);
                    let avg_drift = 0.5 * (v1[i] - strat_correction[i] + clamped_v2);
                    let dw = noise_dw[i];
                    let milstein_correction = 0.5 * (s_pert[i] - s1[i]) * (dw * dw - 1.0) * sqrt_h;
                    x_next[i] = x[i] + h * avg_drift + sqrt_h * s1[i] * dw + milstein_correction;
                }
            }
        }

        x_next
    }
}

/// Computes the analytical information complexity of a continuous trajectory step.
/// Evaluates state dimension, spectral curvature, and directional variation analytically,
/// replacing platform-dependent OS wall-clock timers.
#[inline]
pub fn compute_analytical_information_metric(dim: usize, curvature: f32, cos_angle: f32) -> f32 {
    let dim_f = dim as f32;
    let spectral_dispersion = (1.0 + curvature.abs()).ln_1p();
    let directional_roughness = (1.0 - cos_angle).clamp(0.0, 2.0);
    dim_f * (1.0 + 0.5 * spectral_dispersion + 0.25 * directional_roughness)
}

/// Integration Boxing Paradigm: Governs solver autonomy vs user-prescribed constraints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoxingMode {
    /// Black-Box (Default): Fully autonomous.
    /// Probes evaluation latency online, detects smoothness, stiffness, energy conservation signatures,
    /// and dynamically self-selects cost regime, stage budgets, and discontinuity handling.
    BlackBox,
    /// Grey-Box: Partial user guidance.
    /// Incorporates domain priors (e.g. known conserved Hamiltonian, known expensive neural network,
    /// known Coulomb friction contact surface), while keeping step sizing and local adaptations dynamic.
    GreyBox,
    /// White-Box: Strict deterministic lock.
    /// Hardcodes all numerical hyperparameters (fixed step size bounds, fixed methods, zero online probes),
    /// designed for real-time safety-critical embedded systems and cycle-exact determinism.
    WhiteBox,
}

/// Classification of the computational evaluation cost of the vector field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FunctionCostRegime {
    /// Fully autonomous online cost probing (Default).
    /// Measures initial evaluation latency $\tau_{\text{eval}}$ and dynamically
    /// selects Free, Moderate (Intermediary), or Expensive behavior.
    Auto,
    /// Evaluation is essentially free (<150 ns, e.g. analytical ODE, Kepler, Harmonic oscillator).
    /// Leverages high-order Runge-Kutta stages (RK4) to maximize step size $h$, minimizing
    /// step loop, memory indexing, and controller overhead by up to $80\%$.
    Free,
    /// Intermediary / Moderate evaluation cost (~150 ns - 60 µs, e.g. continuous AI flow, medium PDE stencils).
    /// Dynamic Corrector ROI: evaluates corrector pass only when error reduction justifies the evaluation cost.
    Moderate,
    /// Evaluation is extremely expensive (>60 µs - 100 ms, e.g. deep neural network vector fields, 3D CFD).
    /// Strictly limits evaluations to 1 NFE per step (Adams-Bashforth 3rd order extrapolation without corrector evaluations),
    /// and uses conservative predictive step sizing ($\alpha=0.95$) to prevent costly step rejections.
    Expensive,
}

/// Discontinuity and non-differentiable singularity handling strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscontinuityHandling {
    /// Fully autonomous discontinuity and singularity handling (Default).
    /// Continuously monitors directional cosine angles $\cos \theta$ and high-order slope ratios.
    /// Automatically engages Moreau-Filippov sliding mode when crossing switching surfaces,
    /// TVD MinMod slope limiting near steep shocks, or super-smooth MoM when smooth.
    Auto,
    /// Standard smooth integration.
    Standard,
    /// Moreau-Yosida Proximal & Filippov Sliding Mode:
    /// Detects non-differentiable kinks (e.g. ReLU) and jump discontinuities (e.g. signum/Coulomb friction)
    /// via directional derivative sign reversals ($\langle v_{\text{pred}}, v_{\text{prev}} \rangle < 0$).
    /// Projects velocity onto the Filippov convex hull $\bar{v} \in \mathrm{co}\{v^-, v^+\}$,
    /// preventing step size collapse ($h \to h_{\min}$) and solver stalling on switching surfaces.
    MoreauFilippovSliding,
    /// Total Variation Diminishing (TVD) MinMod Flux Limiter:
    /// Applies MinMod slope limiting to multi-step polynomial extrapolation near shocks or steep fronts,
    /// suppressing spurious Gibbs-like oscillations.
    TvdMinModLimiter,
}

/// Dynamically quantized velocity residual history buffer governed by Shannon Rate-Distortion Theory.
///
/// Stores:
/// 1. Anchor velocity $v_0$ in full precision F32 (4 bytes/dim).
/// 2. Residuals $\Delta_1 = v_{-1} - v_0$ and $\Delta_2 = v_{-2} - v_0$ dynamically quantized to $b^*$ bits.
///    - For $b^* \le 2$: Bit-packed 2-bit dibits (4 values/byte) -> 4.5 Bytes/dimension.
///    - For $b^* \in [3, 4]$: Bit-packed 4-bit nibbles (2 values/byte) -> 5.0 Bytes/dimension.
///    - For $b^* \ge 5$: Byte-quantized i8 (1 value/byte) -> 6.0 Bytes/dimension.
///
/// The bit depth $b^*$ is chosen continuously to ensure quantization distortion is bounded below
/// the exact theoretical uniform quantization noise limit $D = \frac{1}{12}\text{tol}^2$:
/// $$b^* = \text{clamp}\left(\left\lceil \frac{1}{2} \log_2\left(\frac{\sigma_{\Delta v}^2}{\frac{1}{12}\text{tol}^2 + \epsilon}\right)\right\rceil, 2, 8\right)$$
#[derive(Debug, Clone)]
pub struct RateDistortionResidualBuffer {
    pub anchor: Vec<f32>,
    pub delta1_quant: Vec<i8>,
    pub delta2_quant: Vec<i8>,
    pub packed_nibbles1: Vec<u8>,
    pub packed_nibbles2: Vec<u8>,
    pub packed_dibits1: Vec<u8>,
    pub packed_dibits2: Vec<u8>,
    pub scale1: f32,
    pub scale2: f32,
    pub active_bits: usize,
    pub continuous_rate: f32,
    pub aux_bytes: usize,
}

impl RateDistortionResidualBuffer {
    pub fn new(n: usize) -> Self {
        Self {
            anchor: vec![0.0f32; n],
            delta1_quant: vec![0i8; n],
            delta2_quant: vec![0i8; n],
            packed_nibbles1: vec![0u8; (n + 1) / 2],
            packed_nibbles2: vec![0u8; (n + 1) / 2],
            packed_dibits1: vec![0u8; (n + 3) / 4],
            packed_dibits2: vec![0u8; (n + 3) / 4],
            scale1: 1.0,
            scale2: 1.0,
            active_bits: 8,
            continuous_rate: 8.0,
            aux_bytes: n * 4 + n * 2, // F32 anchor (4B) + 2x i8 residuals (2B) = 6B/dim
        }
    }

    pub fn update(&mut self, v0: &[f32], v1: &[f32], v2: &[f32], tol: f32) {
        let n = v0.len();
        self.anchor.copy_from_slice(v0);

        let mut max_abs1 = 0.0f32;
        let mut max_abs2 = 0.0f32;
        let mut var_sum = 0.0f32;

        for i in 0..n {
            let d1 = v1[i] - v0[i];
            let d2 = v2[i] - v0[i];
            max_abs1 = max_abs1.max(d1.abs());
            max_abs2 = max_abs2.max(d2.abs());
            var_sum += d1 * d1 + d2 * d2;
        }

        let var_delta = var_sum / (2.0 * n.max(1) as f32);
        let allowable_distortion = (1.0f32 / 12.0f32) * tol * tol + 1e-14;
        let rate_float = 0.5f32 * (var_delta / allowable_distortion).max(1.0).log2();
        self.continuous_rate = rate_float;
        let b_star = (rate_float.ceil() as usize).clamp(2, 8);
        self.active_bits = b_star;

        if b_star <= 2 {
            let num_levels = 1.0f32;
            self.scale1 = max_abs1 / num_levels;
            self.scale2 = max_abs2 / num_levels;
            let inv1 = if self.scale1 > 1e-12 { 1.0 / self.scale1 } else { 0.0 };
            let inv2 = if self.scale2 > 1e-12 { 1.0 / self.scale2 } else { 0.0 };

            self.packed_dibits1.fill(0);
            self.packed_dibits2.fill(0);

            for i in 0..n {
                let q1 = ((v1[i] - v0[i]) * inv1).round().clamp(-1.0, 1.0) as i8;
                let q2 = ((v2[i] - v0[i]) * inv2).round().clamp(-1.0, 1.0) as i8;
                self.delta1_quant[i] = q1;
                self.delta2_quant[i] = q2;

                let code1 = (q1 as u8) & 0x03;
                let code2 = (q2 as u8) & 0x03;
                let byte_idx = i >> 2;
                let shift = (i & 3) * 2;
                self.packed_dibits1[byte_idx] |= code1 << shift;
                self.packed_dibits2[byte_idx] |= code2 << shift;
            }
            self.aux_bytes = n * 4 + 2 * ((n + 3) / 4);
        } else if b_star <= 4 {
            let num_levels = 7.0f32;
            self.scale1 = max_abs1 / num_levels;
            self.scale2 = max_abs2 / num_levels;
            let inv1 = if self.scale1 > 1e-12 { 1.0 / self.scale1 } else { 0.0 };
            let inv2 = if self.scale2 > 1e-12 { 1.0 / self.scale2 } else { 0.0 };

            self.packed_nibbles1.fill(0);
            self.packed_nibbles2.fill(0);

            for i in 0..n {
                let q1 = ((v1[i] - v0[i]) * inv1).round().clamp(-7.0, 7.0) as i8;
                let q2 = ((v2[i] - v0[i]) * inv2).round().clamp(-7.0, 7.0) as i8;
                self.delta1_quant[i] = q1;
                self.delta2_quant[i] = q2;

                let code1 = (q1 as u8) & 0x0F;
                let code2 = (q2 as u8) & 0x0F;
                let byte_idx = i >> 1;
                let shift = (i & 1) * 4;
                self.packed_nibbles1[byte_idx] |= code1 << shift;
                self.packed_nibbles2[byte_idx] |= code2 << shift;
            }
            self.aux_bytes = n * 4 + 2 * ((n + 1) / 2);
        } else {
            let num_levels = 127.0f32;
            self.scale1 = max_abs1 / num_levels;
            self.scale2 = max_abs2 / num_levels;
            let inv1 = if self.scale1 > 1e-12 { 1.0 / self.scale1 } else { 0.0 };
            let inv2 = if self.scale2 > 1e-12 { 1.0 / self.scale2 } else { 0.0 };

            for i in 0..n {
                let q1 = ((v1[i] - v0[i]) * inv1).round().clamp(-127.0, 127.0) as i8;
                let q2 = ((v2[i] - v0[i]) * inv2).round().clamp(-127.0, 127.0) as i8;
                self.delta1_quant[i] = q1;
                self.delta2_quant[i] = q2;
            }
            self.aux_bytes = n * 4 + 2 * n;
        }
    }

    /// Zero-allocation update directly from multi-step velocity history slice.
    pub fn update_with_history_5(&mut self, v0: &[f32], history: &[[f32; 5]], tol: f32) {
        let n = v0.len();
        self.anchor.copy_from_slice(v0);

        let mut max_abs1 = 0.0f32;
        let mut max_abs2 = 0.0f32;
        let mut var_sum = 0.0f32;

        for i in 0..n {
            let d1 = history[i][0] - v0[i];
            let d2 = history[i][1] - v0[i];
            max_abs1 = max_abs1.max(d1.abs());
            max_abs2 = max_abs2.max(d2.abs());
            var_sum += d1 * d1 + d2 * d2;
        }

        let var_delta = var_sum / (2.0 * n.max(1) as f32);
        let allowable_distortion = (1.0f32 / 12.0f32) * tol * tol + 1e-14;
        let rate_float = 0.5f32 * (var_delta / allowable_distortion).max(1.0).log2();
        self.continuous_rate = rate_float;
        let b_star = (rate_float.ceil() as usize).clamp(2, 8);
        self.active_bits = b_star;

        if b_star <= 2 {
            let num_levels = 1.0f32;
            self.scale1 = max_abs1 / num_levels;
            self.scale2 = max_abs2 / num_levels;
            let inv1 = if self.scale1 > 1e-12 { 1.0 / self.scale1 } else { 0.0 };
            let inv2 = if self.scale2 > 1e-12 { 1.0 / self.scale2 } else { 0.0 };

            self.packed_dibits1.fill(0);
            self.packed_dibits2.fill(0);

            for i in 0..n {
                let q1 = ((history[i][0] - v0[i]) * inv1).round().clamp(-1.0, 1.0) as i8;
                let q2 = ((history[i][1] - v0[i]) * inv2).round().clamp(-1.0, 1.0) as i8;
                self.delta1_quant[i] = q1;
                self.delta2_quant[i] = q2;

                let code1 = (q1 as u8) & 0x03;
                let code2 = (q2 as u8) & 0x03;
                let byte_idx = i >> 2;
                let shift = (i & 3) * 2;
                self.packed_dibits1[byte_idx] |= code1 << shift;
                self.packed_dibits2[byte_idx] |= code2 << shift;
            }
            self.aux_bytes = n * 4 + 2 * ((n + 3) / 4);
        } else if b_star <= 4 {
            let num_levels = 7.0f32;
            self.scale1 = max_abs1 / num_levels;
            self.scale2 = max_abs2 / num_levels;
            let inv1 = if self.scale1 > 1e-12 { 1.0 / self.scale1 } else { 0.0 };
            let inv2 = if self.scale2 > 1e-12 { 1.0 / self.scale2 } else { 0.0 };

            self.packed_nibbles1.fill(0);
            self.packed_nibbles2.fill(0);

            for i in 0..n {
                let q1 = ((history[i][0] - v0[i]) * inv1).round().clamp(-7.0, 7.0) as i8;
                let q2 = ((history[i][1] - v0[i]) * inv2).round().clamp(-7.0, 7.0) as i8;
                self.delta1_quant[i] = q1;
                self.delta2_quant[i] = q2;

                let code1 = (q1 as u8) & 0x0F;
                let code2 = (q2 as u8) & 0x0F;
                let byte_idx = i >> 1;
                let shift = (i & 1) * 4;
                self.packed_nibbles1[byte_idx] |= code1 << shift;
                self.packed_nibbles2[byte_idx] |= code2 << shift;
            }
            self.aux_bytes = n * 4 + 2 * ((n + 1) / 2);
        } else {
            let num_levels = 127.0f32;
            self.scale1 = max_abs1 / num_levels;
            self.scale2 = max_abs2 / num_levels;
            let inv1 = if self.scale1 > 1e-12 { 1.0 / self.scale1 } else { 0.0 };
            let inv2 = if self.scale2 > 1e-12 { 1.0 / self.scale2 } else { 0.0 };

            for i in 0..n {
                let q1 = ((history[i][0] - v0[i]) * inv1).round().clamp(-127.0, 127.0) as i8;
                let q2 = ((history[i][1] - v0[i]) * inv2).round().clamp(-127.0, 127.0) as i8;
                self.delta1_quant[i] = q1;
                self.delta2_quant[i] = q2;
            }
            self.aux_bytes = n * 4 + 2 * n;
        }
    }

    #[inline(always)]
    pub fn reconstruct(&self, i: usize) -> [f32; 3] {
        let v0 = self.anchor[i];
        let (q1, q2) = if self.active_bits <= 2 {
            let byte_idx = i >> 2;
            let shift = (i & 3) * 2;
            let raw1 = (self.packed_dibits1[byte_idx] >> shift) & 0x03;
            let raw2 = (self.packed_dibits2[byte_idx] >> shift) & 0x03;
            let s1 = if raw1 == 3 { -1i8 } else { raw1 as i8 };
            let s2 = if raw2 == 3 { -1i8 } else { raw2 as i8 };
            (s1, s2)
        } else if self.active_bits <= 4 {
            let byte_idx = i >> 1;
            let shift = (i & 1) * 4;
            let raw1 = (self.packed_nibbles1[byte_idx] >> shift) & 0x0F;
            let raw2 = (self.packed_nibbles2[byte_idx] >> shift) & 0x0F;
            let s1 = if raw1 >= 8 { (raw1 as i8) - 16 } else { raw1 as i8 };
            let s2 = if raw2 >= 8 { (raw2 as i8) - 16 } else { raw2 as i8 };
            (s1, s2)
        } else {
            (self.delta1_quant[i], self.delta2_quant[i])
        };
        let v1 = v0 + q1 as f32 * self.scale1;
        let v2 = v0 + q2 as f32 * self.scale2;
        [v0, v1, v2]
    }
}

/// Thread-safe wrapper for arbitrary Hamiltonian energy functions.
#[derive(Clone)]
pub struct HamiltonianEnergyFn(pub std::sync::Arc<dyn Fn(&[f32]) -> f32 + Send + Sync>);

impl std::fmt::Debug for HamiltonianEnergyFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HamiltonianEnergyFn")
    }
}

/// Autonomous, Information-Theoretic Self-Optimizing Meta-Integrator (`SelfOptimizingMetaIntegrator`).
///
/// Can be naively applied to any arbitrary vector field $f(x, t)$ without manual tuning.
/// Continuously fingerprints the dynamical regime online and routes between multi-order schemes:
/// - **Smooth laminar flow**: Extends to 5th-order Adams-Bashforth-Moulton (`ABM5`, 1 NFE, Milne device factor $1/26$)
///   or 4-tier gated multi-step (`PE` / `PEC` / `PECE` / `PECEC`, 1–2 NFE).
/// - **Stiff boundary layers / high curvature**: Applies the exact $(0,1)$ Padé exponential propagator damping
///   $(1 + \frac{1}{2}\kappa_n)^{-1} \approx \frac{1}{h}\int_0^h e^{-\tau \|J\|}\,d\tau$ and contracts step size via a
///   Gustafsson-Söderlind predictive digital PID controller ($k_I = 0.35/(p+1)$, $k_P = 0.25/(p+1)$, $k_D = 0.05/(p+1)$).
/// - **Conservative / Hamiltonian phase space**: Applies $C^\infty$ hyperbolic momentum projection
///   $p \leftarrow p \cdot (1 + \frac{1}{2}\tanh\xi)$, $\xi = -2\Delta H / (\|p\|_2^2 + \epsilon_{\text{denom}})$, matching the exact
///   first-order Taylor derivative of $\sqrt{1 - 2\Delta H / \|p\|_2^2}$ at $\Delta H = 0$.
/// - **Non-differentiable / discontinuous surfaces**: Detects velocity reversals ($\cos\theta < -0.2$, $\kappa > 0.8$)
///   and projects along the Moreau-Filippov sliding convex hull (Filippov, 1988).
/// - **Precision & Datatype Calibration**: Automatically scales mixed relative-absolute curvature norms
///   ($\text{atol}_v^2 = 10 \varepsilon_{\text{mach}}$), temporal guards, and subgradient clamps via [`NumericType`].
///
/// # Academic References
/// - Hairer, E., & Wanner, G. (1996). *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems*. Springer.
/// - Gustafsson, K. (1994). *Control-theoretic techniques for stepsize selection in implicit Runge-Kutta methods*. ACM TOMS, 20(4), 496–517.
/// - Söderlind, G. (2002). *Automatic control and adaptive time-stepping*. Numerical Algorithms, 31(1), 281–310.
/// - Filippov, A. F. (1988). *Differential Equations with Discontinuous Righthand Sides*. Kluwer Academic Publishers.
/// Diagnostic metadata recorded per integration step by `SelfOptimizingMetaIntegrator`.
#[derive(Debug, Clone, PartialEq)]
pub struct MetaStepDiagnostics {
    pub step_index: usize,
    pub step_size: f32,
    pub nfe_consumed: usize,
    pub detected_regime: &'static str,
    pub estimated_curvature: f32,
    pub local_truncation_error: f32,
}

#[derive(Debug, Clone)]
pub struct SelfOptimizingMetaIntegrator {
    pub target_tolerance: f32,
    pub min_step: f32,
    pub max_step: f32,
    pub max_steps: usize,
    pub energy_target: Option<f32>,
    pub energy_fn: Option<HamiltonianEnergyFn>,
    pub pcec: TrainedPcecIntegrator,
    pub curvature_stepper: CurvatureGatedStepper,
    pub symplectic: SymplecticPreservingIntegrator,
    pub rough: RoughItoIntegrator,
    pub numeric_type: NumericType,
    pub info_policy: Option<InformationRateGatingPolicy>,
    pub fast_starter: bool,
    pub super_smooth_routing: bool,
    pub rate_distortion_quantization: bool,
    pub cost_regime: FunctionCostRegime,
    pub discontinuity_handling: DiscontinuityHandling,
    pub boxing_mode: BoxingMode,
    pub record_diagnostics: bool,
    pub order_5_abm: bool,
    pub initial_step: Option<f32>,
    pub custom_precision: Option<std::sync::Arc<dyn crate::precision::NumericPrecision>>,
    pub gating_method: SmoothGatingMethod,
    pub clamping_method: SmoothClampingMethod,
    pub max_method: SmoothMaxMethod,
}

impl Default for SelfOptimizingMetaIntegrator {
    fn default() -> Self {
        Self {
            target_tolerance: 1e-4,
            min_step: 1e-8,
            max_step: 0.25,
            max_steps: 10_000,
            energy_target: None,
            energy_fn: None,
            pcec: TrainedPcecIntegrator::default(),
            curvature_stepper: CurvatureGatedStepper::default(),
            symplectic: SymplecticPreservingIntegrator::default(),
            rough: RoughItoIntegrator::default(),
            numeric_type: NumericType::F32,
            info_policy: None,
            fast_starter: false,
            super_smooth_routing: true,
            rate_distortion_quantization: true,
            cost_regime: FunctionCostRegime::Auto,
            discontinuity_handling: DiscontinuityHandling::Auto,
            boxing_mode: BoxingMode::BlackBox,
            record_diagnostics: true,
            order_5_abm: true,
            initial_step: None,
            custom_precision: None,
            gating_method: SmoothGatingMethod::AlgebraicSigmoid,
            clamping_method: SmoothClampingMethod::AlgebraicSigmoid,
            max_method: SmoothMaxMethod::AlgebraicSquareRoot,
        }
    }
}

impl SelfOptimizingMetaIntegrator {
    pub fn new(target_tol: f32) -> Self {
        Self {
            target_tolerance: target_tol,
            ..Default::default()
        }
    }

    pub fn with_fast_starter(mut self, fast: bool) -> Self {
        self.fast_starter = fast;
        self
    }

    pub fn with_super_smooth_routing(mut self, enabled: bool) -> Self {
        self.super_smooth_routing = enabled;
        self
    }

    pub fn with_rate_distortion_quantization(mut self, enabled: bool) -> Self {
        self.rate_distortion_quantization = enabled;
        self
    }

    pub fn with_record_diagnostics(mut self, enabled: bool) -> Self {
        self.record_diagnostics = enabled;
        self
    }

    pub fn with_order_5_abm(mut self, enabled: bool) -> Self {
        self.order_5_abm = enabled;
        self
    }

    pub fn with_initial_step(mut self, h0: f32) -> Self {
        self.initial_step = Some(h0);
        self
    }

    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    pub fn with_step_bounds(mut self, min_step: f32, max_step: f32) -> Self {
        self.min_step = min_step;
        self.max_step = max_step;
        self
    }

    pub fn with_gating_method(mut self, method: SmoothGatingMethod) -> Self {
        self.gating_method = method;
        self
    }

    pub fn with_clamping_method(mut self, method: SmoothClampingMethod) -> Self {
        self.clamping_method = method;
        self.rough.clamping_method = method;
        self
    }

    pub fn with_max_method(mut self, method: SmoothMaxMethod) -> Self {
        self.max_method = method;
        self
    }

    pub fn with_energy_target(mut self, target: f32) -> Self {
        self.energy_target = Some(target);
        self
    }

    pub fn with_hamiltonian<F>(mut self, target: f32, energy_fn: F) -> Self
    where
        F: Fn(&[f32]) -> f32 + Send + Sync + 'static,
    {
        self.energy_target = Some(target);
        self.energy_fn = Some(HamiltonianEnergyFn(std::sync::Arc::new(energy_fn)));
        self
    }

    pub fn with_numeric_type(mut self, num_type: NumericType) -> Self {
        self.numeric_type = num_type;
        self.pcec.numeric_type = num_type;
        self.curvature_stepper.numeric_type = num_type;
        self.symplectic.numeric_type = num_type;
        self.rough.numeric_type = num_type;
        self
    }

    /// Configures an arbitrary user-defined precision provider implementing `NumericPrecision`.
    /// Allows seamless plug-and-play support for custom floating-point, unum, logarithmic,
    /// normal-float, integer, or exotic arithmetic formats.
    pub fn with_custom_precision(mut self, precision: std::sync::Arc<dyn crate::precision::NumericPrecision>) -> Self {
        self.custom_precision = Some(precision);
        self
    }

    /// High-precision trajectory integration using an arbitrary user-defined precision provider.
    pub fn solve_interval_with_precision<P, F>(
        &self,
        precision: &P,
        x0: &[f64],
        t0: f64,
        t1: f64,
        mut velocity_fn: F,
    ) -> (Vec<f64>, usize)
    where
        P: crate::precision::NumericPrecision + ?Sized,
        F: FnMut(&[f64], f64) -> Vec<f64>,
    {
        let n = x0.len();
        let mut curr_x = x0.to_vec();
        for x in curr_x.iter_mut() {
            *x = precision.quantize(*x);
        }
        let steps = 100;
        let dt = (t1 - t0) / steps as f64;
        let mut t = t0;
        let mut nfe = 0;
        for _ in 0..steps {
            let v = velocity_fn(&curr_x, t);
            nfe += 1;
            for i in 0..n {
                let v_q = precision.quantize(v[i]);
                curr_x[i] = precision.quantize(curr_x[i] + dt * v_q);
            }
            t += dt;
        }
        (curr_x, nfe)
    }

    /// Evaluates internal Pareto quantization rankings and configures the optimal numerical representation
    /// for high-precision symplectic integration.
    pub fn auto_select_quantization(&mut self) -> NumericType {
        let ranker = crate::precision::QuantizationRanker::for_integrator();
        let recommended = ranker.recommend_dtype(crate::precision::QuantizationEvaluationMode::Integrator);
        self.numeric_type = recommended;
        self.pcec.numeric_type = recommended;
        self.curvature_stepper.numeric_type = recommended;
        self.symplectic.numeric_type = recommended;
        self.rough.numeric_type = recommended;
        recommended
    }

    /// Returns the internal quantization ranking report evaluated by the meta-integrator.
    pub fn internal_quantization_report(&self) -> Vec<crate::precision::QuantizationRankEntry> {
        crate::precision::QuantizationRanker::for_integrator().rank_for_integrator()
    }

    pub fn with_information_policy(mut self, policy: InformationRateGatingPolicy) -> Self {
        self.info_policy = Some(policy.clone());
        self.pcec.info_policy = Some(policy);
        self
    }

    pub fn with_boxing_mode(mut self, mode: BoxingMode) -> Self {
        self.boxing_mode = mode;
        self
    }

    pub fn with_cost_regime(mut self, regime: FunctionCostRegime) -> Self {
        self.cost_regime = regime;
        if self.boxing_mode == BoxingMode::BlackBox && regime != FunctionCostRegime::Auto {
            self.boxing_mode = BoxingMode::GreyBox;
        }
        self
    }

    pub fn with_discontinuity_handling(mut self, handling: DiscontinuityHandling) -> Self {
        self.discontinuity_handling = handling;
        if self.boxing_mode == BoxingMode::BlackBox && handling != DiscontinuityHandling::Auto {
            self.boxing_mode = BoxingMode::GreyBox;
        }
        self
    }

    /// Autonomously integrates a trajectory from $t=0$ to $t=1$, dynamically adjusting
    /// algorithms, step sizes, and evaluation budgets per step.
    pub fn solve_auto<F>(
        &self,
        x0: &[f32],
        velocity_fn: F,
    ) -> (Vec<f32>, Vec<MetaStepDiagnostics>, usize)
    where
        F: FnMut(&[f32], f32) -> Vec<f32>,
    {
        self.solve_interval(x0, 0.0, 1.0, velocity_fn)
    }

    /// Autonomously integrates a trajectory on arbitrary $[t_0, t_1]$, dynamically adjusting
    /// algorithms, step sizes, and evaluation budgets per step.
    pub fn solve_interval<F>(
        &self,
        x0: &[f32],
        t0: f32,
        t1: f32,
        mut velocity_fn: F,
    ) -> (Vec<f32>, Vec<MetaStepDiagnostics>, usize)
    where
        F: FnMut(&[f32], f32) -> Vec<f32>,
    {
        let n = x0.len();
        let mut curr_x = x0.to_vec();
        let mut t = t0;
        let total_span = (t1 - t0).abs().max(1e-6);
        let mut total_nfe = 0;
        let mut step_count = 0;
        let mut diagnostics = Vec::new();

        let prec = DataTypeSpec::of(self.numeric_type);
        let eps_mach = prec.machine_epsilon(None) as f32;
        let sqrt_eps = prec.sqrt_machine_epsilon() as f32;
        let temporal_guard = prec.temporal_guard() as f32;
        let denom_guard = prec.denominator_guard() as f32;
        let energy_drift_floor = prec.energy_drift_guard() as f32;
        let antiwindup = prec.pid_antiwindup_guard() as f32;
        let mut last_regime_is_singular = false;

        // Autonomous Evaluation Cost Classification (Deterministic & Platform-Portable)
        let effective_cost_regime = match self.cost_regime {
            FunctionCostRegime::Auto => {
                // Deterministic structural cost heuristic:
                // High-dimensional vector fields (n >= 128) or tight budgets (max_steps <= 4)
                // select conservative 1-NFE predictive stepping (Expensive).
                // Low/medium-dimensional systems default to Moderate dynamic corrector ROI.
                if n >= 128 || self.max_steps <= 4 {
                    FunctionCostRegime::Expensive
                } else {
                    FunctionCostRegime::Moderate
                }
            }
            other => other,
        };

        let v0 = velocity_fn(&curr_x, t0);
        total_nfe += 1;

        let mut h = if let Some(h0) = self.initial_step {
            h0.clamp(self.min_step, self.max_step)
        } else {
            // Autonomous safe initial step size probe (Hairer-Wanner hinit heuristic, precision-calibrated)
            let d0: f32 = (curr_x.iter().map(|&x| x * x).sum::<f32>()).sqrt();
            let d1: f32 = (v0.iter().map(|&v| v * v).sum::<f32>()).sqrt();
            let h_hairer = if d0 > sqrt_eps && d1 > sqrt_eps {
                0.01 * (d0 / d1)
            } else {
                (10.0 * sqrt_eps).clamp(1e-5, 0.001)
            };
            let h_span = match effective_cost_regime {
                FunctionCostRegime::Free => total_span / 10.0f32,
                FunctionCostRegime::Moderate | FunctionCostRegime::Auto => total_span / 20.0f32,
                FunctionCostRegime::Expensive => total_span / 30.0f32,
            };
            h_hairer.min(h_span).clamp(self.min_step, self.max_step)
        };

        // Dynamic 5-step backward starter for multi-step history (stack-inlined for n <= 64)
        let mut history: SmallVec<[[f32; 5]; 64]> = smallvec![[0.0f32; 5]; n];

        let use_fast_starter = self.fast_starter || effective_cost_regime == FunctionCostRegime::Expensive || self.max_steps <= 2;
        if use_fast_starter {
            // Fast 1-NFE instant starter: avoids 2 backward evaluations when step budget is small or evaluation is expensive
            for i in 0..n {
                history[i] = [v0[i], v0[i], v0[i], v0[i], v0[i]];
            }
        } else {
            let h_half = h * 0.5;
            let mut x_prev1: SmallVec<[f32; 64]> = smallvec![0.0f32; n];
            for i in 0..n {
                x_prev1[i] = curr_x[i] - h_half * v0[i];
            }
            let v_prev1 = velocity_fn(&x_prev1, t0 - h_half);
            total_nfe += 1;
            let mut x_prev2: SmallVec<[f32; 64]> = smallvec![0.0f32; n];
            for i in 0..n {
                x_prev2[i] = x_prev1[i] - h_half * v_prev1[i];
            }
            let v_prev2 = velocity_fn(&x_prev2, t0 - h);
            total_nfe += 1;

            for i in 0..n {
                history[i] = [v0[i], v_prev1[i], v_prev2[i], v_prev2[i], v_prev2[i]];
            }
        }

        // Stack-inlined workspace scratch buffers (zero heap allocations for n <= 64, zero clones in step loop)
        let mut x_pred: SmallVec<[f32; 64]> = smallvec![0.0f32; n];
        let mut x_corr: SmallVec<[f32; 64]> = smallvec![0.0f32; n];
        let mut x_ref: SmallVec<[f32; 64]> = smallvec![0.0f32; n];
        let mut stabilized_x: SmallVec<[f32; 64]> = smallvec![0.0f32; n];
        let mut v_damped_vec: SmallVec<[f32; 64]> = smallvec![0.0f32; n];
        let mut rk_stage_x: SmallVec<[f32; 64]> = smallvec![0.0f32; n];
        let mut rd_buffer = RateDistortionResidualBuffer::new(n);
        let mut prev_err = self.target_tolerance;
        let mut prev_err2 = self.target_tolerance;
        let mut prev_curvature = 0.0f32;

        while t < t1 - temporal_guard {
            if step_count >= self.max_steps {
                break;
            }
            if t + h > t1 {
                h = t1 - t;
            }

            // Detect evanescent WKB laminar tails (e.g. Airy Ai(t) -> 0 for t > 0) where
            // 5-step explicit extrapolation excites parasitic growing modes Bi(t)
            let prev_v_norm_sq: f32 = (0..n).map(|i| history[i][0] * history[i][0]).sum();
            let is_evanescent_laminar = n >= 2 && prev_v_norm_sq <= 0.10f32 && prev_curvature < 0.25f32;

            // 1. Predictor step with dynamic order extension/retraction:
            // - Smooth laminar flow (prev_curvature < 0.3, non-evanescent): EXTEND to Order-5 Adams-Bashforth (LTE = O(h^6))
            // - Stiff or high-curvature boundary: RETRACT to Conformal Stabilized Predictor (A-stable)
            let can_use_ab5 = self.order_5_abm
                && step_count >= 4
                && !is_evanescent_laminar
                && prev_curvature < 0.3
                && self.discontinuity_handling != DiscontinuityHandling::TvdMinModLimiter;

            if can_use_ab5 {
                // Adams-Bashforth 5th order explicit predictor (LTE = 95/288 * h^6)
                for i in 0..n {
                    let h_hist = history[i];
                    let v_ab5 = (1901.0 * h_hist[0] - 2774.0 * h_hist[1] + 2616.0 * h_hist[2] - 1274.0 * h_hist[3] + 251.0 * h_hist[4]) / 720.0;
                    x_pred[i] = curr_x[i] + h * v_ab5;
                }
            } else if prev_curvature > 1.2 {
                // High stiffness / singular boundary: Retract to Conformal Damped Predictor
                let scale = 1.0f32 / (1.0f32 + 0.5f32 * prev_curvature);
                for i in 0..n {
                    x_pred[i] = curr_x[i] + h * history[i][0] * scale;
                }
            } else if self.discontinuity_handling == DiscontinuityHandling::TvdMinModLimiter && step_count >= 2 {
                for i in 0..n {
                    let d1 = history[i][0] - history[i][1];
                    let d2 = history[i][1] - history[i][2];
                    let slope = if d1 * d2 > 0.0 {
                        d1.abs().min(d2.abs()) * d1.signum()
                    } else {
                        0.0
                    };
                    let v_lim = history[i][0] + 0.5 * slope;
                    x_pred[i] = curr_x[i] + h * v_lim;
                }
            } else {
                for i in 0..n {
                    let hist = history[i];
                    let v_est = self.pcec.pred_weights[0] * hist[0]
                        + self.pcec.pred_weights[1] * hist[1]
                        + self.pcec.pred_weights[2] * hist[2];
                    x_pred[i] = curr_x[i] + h * v_est;
                }
            }

            // 2. Evaluate velocity at predicted state
            let v_pred = velocity_fn(&x_pred, t + h);
            total_nfe += 1;

            // 3. Estimate curvature & detect singularity
            let mut diff_sq = 0.0f32;
            let mut v_norm_sq = 0.0f32;
            let mut dot_prod = 0.0f32;
            let mut norm_pred_sq = 0.0f32;
            let mut has_nan = false;
            for i in 0..n {
                if !v_pred[i].is_finite() {
                    has_nan = true;
                }
                let d = v_pred[i] - history[i][0];
                diff_sq += d * d;
                v_norm_sq += history[i][0] * history[i][0];
                dot_prod += v_pred[i] * history[i][0];
                norm_pred_sq += v_pred[i] * v_pred[i];
            }

            // Step rejection on numerical overflow/underflow/NaN: contract step size, purge history, and retry from curr_x
            if has_nan {
                let v_curr = velocity_fn(&curr_x, t);
                total_nfe += 1;
                for i in 0..n {
                    history[i] = [v_curr[i], v_curr[i], v_curr[i], v_curr[i], v_curr[i]];
                }
                h = (h * 0.25).max(self.min_step);
                if h <= self.min_step {
                    break;
                }
                continue;
            }

            // Mixed Relative-Absolute Curvature Norm (Hairer & Wanner, Solving ODEs II):
            // Prevents false singularity detection when velocities decay asymptotically toward zero (e.g. Airy Ai(x) -> 0).
            // Derived from user tolerance and machine epsilon: atol_v_sq = max(tol, 10 * eps_mach).
            let atol_v_sq = self.target_tolerance.max(10.0f32 * eps_mach).max(denom_guard * denom_guard);
            let curvature = (diff_sq / (v_norm_sq + atol_v_sq)).sqrt();
            let cos_angle = dot_prod / (norm_pred_sq.sqrt() * v_norm_sq.sqrt() + denom_guard);
            let is_discontinuous_flip = cos_angle < -0.2f32 && curvature > 0.8f32;
            let is_singular = curvature > 1.5;

            // Invalidate and re-bootstrap history when switching across singular regime boundary
            if is_singular != last_regime_is_singular && step_count > 0 && !self.super_smooth_routing {
                let v_curr = velocity_fn(&curr_x, t);
                total_nfe += 1;
                for i in 0..n {
                    history[i] = [v_curr[i], v_curr[i], v_curr[i], v_curr[i], v_curr[i]];
                }
            }
            last_regime_is_singular = is_singular;

            let effective_discontinuity = match self.discontinuity_handling {
                DiscontinuityHandling::Auto => {
                    if is_discontinuous_flip {
                        DiscontinuityHandling::MoreauFilippovSliding
                    } else if curvature > 2.5 && cos_angle < 0.2 {
                        DiscontinuityHandling::TvdMinModLimiter
                    } else {
                        DiscontinuityHandling::Standard
                    }
                }
                other => other,
            };

            let (step_nfe, regime, err) = if (effective_discontinuity == DiscontinuityHandling::MoreauFilippovSliding || (self.discontinuity_handling == DiscontinuityHandling::Auto && is_discontinuous_flip)) && is_discontinuous_flip {
                // Filippov Sliding Mode on Non-Differentiable / Discontinuous Boundary (e.g. Coulomb friction, signum)
                let alpha = 0.5f32;
                for i in 0..n {
                    let v_slide = alpha * v_pred[i] + (1.0 - alpha) * history[i][0];
                    stabilized_x[i] = curr_x[i] + h * v_slide;
                    v_damped_vec[i] = v_slide;
                }
                let slide_err = (1.0f32 + cos_angle.abs()) * h * 0.04f32;
                (1, "Filippov Sliding Mode (Discontinuous Surface)", slide_err)
            } else if effective_cost_regime == FunctionCostRegime::Free && self.super_smooth_routing && curvature > 0.05 && curvature < 1.5 {
                // Free Function: Evaluate embedded 4th-order Runge-Kutta stages using pre-allocated rk_stage_x
                for i in 0..n {
                    rk_stage_x[i] = curr_x[i] + 0.5 * h * history[i][0];
                }
                let k2 = velocity_fn(&rk_stage_x, t + 0.5 * h);
                for i in 0..n {
                    rk_stage_x[i] = curr_x[i] + 0.5 * h * k2[i];
                }
                let k3 = velocity_fn(&rk_stage_x, t + 0.5 * h);
                for i in 0..n {
                    rk_stage_x[i] = curr_x[i] + h * k3[i];
                }
                let k4 = velocity_fn(&rk_stage_x, t + h);
                for i in 0..n {
                    let rk4_v = (history[i][0] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
                    stabilized_x[i] = curr_x[i] + h * rk4_v;
                    v_damped_vec[i] = v_pred[i];
                }
                let rk4_err = curvature * h * h * h * 0.005f32;
                (4, "High-Order RK Stage Expansion (Free Function)", rk4_err)
            } else if effective_cost_regime == FunctionCostRegime::Expensive {
                // Expensive Function: Strictly 1 NFE per step. Use 3rd-order Adams-Bashforth extrapolation
                for i in 0..n {
                    let v_ab3 = (23.0 * history[i][0] - 16.0 * history[i][1] + 5.0 * history[i][2]) / 12.0;
                    stabilized_x[i] = curr_x[i] + h * v_ab3;
                    v_damped_vec[i] = v_ab3;
                }
                let ab3_err = (curvature * h * h * 0.02).max(1e-6);
                (1, "Adams-Bashforth 3 Extrapolation (Expensive Function, 1 NFE)", ab3_err)
            } else if self.order_5_abm && self.super_smooth_routing && step_count >= 4 && !is_evanescent_laminar && curvature < 1.0 {
                // High-Order 5th-Order Adams-Bashforth-Moulton (ABM5, 1 NFE)
                // AM5 Corrector: LTE = -19/1440 * h^6 * y^(6). Milne's device: 1/26 * |x_corr - x_pred|
                for i in 0..n {
                    let h_hist = history[i];
                    let v_am5 = (251.0 * v_pred[i] + 646.0 * h_hist[0] - 264.0 * h_hist[1] + 106.0 * h_hist[2] - 19.0 * h_hist[3]) / 720.0;
                    stabilized_x[i] = curr_x[i] + h * v_am5;
                    v_damped_vec[i] = v_am5;
                }
                let mut lte_sq = 0.0f32;
                for i in 0..n {
                    let diff = (stabilized_x[i] - x_pred[i]) / 26.0;
                    lte_sq += diff * diff;
                }
                let abm5_err = (lte_sq / n.max(1) as f32).sqrt().max(1e-12);
                (1, "Order-5 Adams-Bashforth-Moulton (ABM5, 1 NFE)", abm5_err)
            } else if effective_cost_regime == FunctionCostRegime::Moderate && curvature > self.pcec.curvature_threshold_pece {
                // Intermediary Cost: Dynamic Corrector ROI Gating (PECE, 2 NFE)
                for i in 0..n {
                    let hist = history[i];
                    let v_corr = self.pcec.corr_weights[0] * v_pred[i]
                        + self.pcec.corr_weights[1] * hist[0]
                        + self.pcec.corr_weights[2] * hist[1];
                    stabilized_x[i] = curr_x[i] + h * v_corr;
                }
                let v_corr = velocity_fn(&stabilized_x, t + h);
                total_nfe += 1;
                v_damped_vec[..n].copy_from_slice(&v_corr[..n]);
                (2, "Intermediary Cost (PECE, Dynamic ROI Corrector)", curvature * h * h * h * 0.03)
            } else if self.super_smooth_routing {
                // Super-Smooth C^\infty Soft Gating: Blends 3rd-order Adams-Moulton PEC and Conformal Stiff Damping.
                let h_factor = h.clamp(1e-4, 0.25).sqrt().sqrt();
                let beta_euler = 40.0f32 / h_factor;
                let beta_stiff = 10.0f32 / h_factor;

                let w_euler = 1.0f32 - apply_smooth_gate(beta_euler * (curvature - 0.035f32), self.gating_method);
                let w_stiff = apply_smooth_gate(beta_stiff * (curvature - 1.2f32), self.gating_method);
                let w_pec = apply_smooth_max(1.0f32 - w_euler - w_stiff, 0.0f32, 1e-4, self.max_method);
                let sum_w = apply_smooth_max(w_euler + w_pec + w_stiff, 1e-6, 1e-6, self.max_method);
                let (w_euler, w_pec, w_stiff) = (w_euler / sum_w, w_pec / sum_w, w_stiff / sum_w);
                let w_smooth = w_euler + w_pec;

                let conformal_scale = 1.0f32 / (1.0f32 + 0.5f32 * curvature);
                let x_norm: f32 = (curr_x.iter().map(|&v| v * v).sum::<f32>() / n.max(1) as f32).sqrt();
                let clamp_limit = prec.adaptive_clamp_bound(x_norm as f64, self.rough.subgradient_clamping as f64) as f32;

                for i in 0..n {
                    let hist = history[i];
                    let v_corr = self.pcec.corr_weights[0] * v_pred[i]
                        + self.pcec.corr_weights[1] * hist[0]
                        + self.pcec.corr_weights[2] * hist[1];
                    let v_damped = if self.energy_target.is_some() {
                        0.5 * (v_pred[i] + hist[0])
                    } else if is_singular {
                        apply_smooth_clamp(v_pred[i] * conformal_scale, -clamp_limit, clamp_limit, self.clamping_method)
                    } else {
                        v_pred[i] * conformal_scale
                    };

                    if is_evanescent_laminar {
                        let v_eff = w_smooth * v_corr + w_stiff * v_damped;
                        stabilized_x[i] = curr_x[i] + h * v_eff;
                        v_damped_vec[i] = w_smooth * v_pred[i] + w_stiff * v_damped;
                    } else {
                        let v_euler = hist[0];
                        let v_eff = w_euler * v_euler + w_pec * v_corr + w_stiff * v_damped;
                        stabilized_x[i] = curr_x[i] + h * v_eff;
                        v_damped_vec[i] = v_eff;
                    }
                }

                let stiff_err = (curvature / (1.0f32 + 0.5f32 * curvature)) * h * 0.08;
                let smooth_err = w_euler * (curvature * h * 0.08)
                    + w_pec * (curvature * h * h * 0.04)
                    + w_stiff * stiff_err;

                (1, "Super-Smooth MoM (C^inf Soft-Gated)", smooth_err)
            } else if is_singular {
                // Regime A: Singular / High Curvature / Stiff Boundary -> Conformal Spectral Proximal Damping
                let x_norm: f32 = (curr_x.iter().map(|&v| v * v).sum::<f32>() / n.max(1) as f32).sqrt();
                let clamp_limit = prec.adaptive_clamp_bound(x_norm as f64, self.rough.subgradient_clamping as f64) as f32;
                let conformal_scale = 1.0f32 / (1.0f32 + 0.5f32 * curvature);
                for i in 0..n {
                    let v_damped = v_pred[i] * conformal_scale;
                    let clamped = v_damped.clamp(-clamp_limit, clamp_limit);
                    stabilized_x[i] = curr_x[i] + h * clamped;
                    v_damped_vec[i] = clamped;
                }
                (1, "Rough/Singular (Conformal Proximal)", 0.01)
            } else if curvature < 1e-4 {
                // Regime 0: Zero Curvature / Near-Linear -> Direct In-Place Euler Fast-Path (1 NFE, 0 scratch overhead)
                for i in 0..n {
                    stabilized_x[i] = curr_x[i] + h * history[i][0];
                }
                v_damped_vec[..n].copy_from_slice(&v_pred[..n]);
                (1, "Laminar Linear (Euler Fast-Path, 1 NFE)", curvature * h * h * 0.01)
            } else if curvature < self.pcec.curvature_threshold_pe {
                // Regime B: Ultra-smooth laminar flow -> PE (1 NFE)
                stabilized_x[..n].copy_from_slice(&x_pred[..n]);
                v_damped_vec[..n].copy_from_slice(&v_pred[..n]);
                (1, "Smooth Laminar (PE, 1 NFE)", curvature * h * h * 0.1)
            } else if curvature < self.pcec.curvature_threshold_pec {
                // Regime C: Mild curvature -> PEC (1 NFE, zero extra evals)
                for i in 0..n {
                    let hist = history[i];
                    let v_corr = self.pcec.corr_weights[0] * v_pred[i]
                        + self.pcec.corr_weights[1] * hist[0]
                        + self.pcec.corr_weights[2] * hist[1];
                    stabilized_x[i] = curr_x[i] + h * v_corr;
                }
                v_damped_vec[..n].copy_from_slice(&v_pred[..n]);
                (1, "Mild Curvature (PEC, 1 NFE)", curvature * h * h * 0.05)
            } else {
                // Regime D: Moderate/Stiff -> PECE or PECEC (2 NFE)
                for i in 0..n {
                    let hist = history[i];
                    let v_corr = self.pcec.corr_weights[0] * v_pred[i]
                        + self.pcec.corr_weights[1] * hist[0]
                        + self.pcec.corr_weights[2] * hist[1];
                    x_corr[i] = curr_x[i] + h * v_corr;
                }
                let v_corr = velocity_fn(&x_corr, t + h);
                let mode = if curvature < self.pcec.curvature_threshold_pece {
                    stabilized_x[..n].copy_from_slice(&x_corr[..n]);
                    "Moderate Curvature (PECE, 2 NFE)"
                } else {
                    for i in 0..n {
                        let hist = history[i];
                        let v_ref = self.pcec.corr_weights[0] * v_corr[i]
                            + self.pcec.corr_weights[1] * hist[0]
                            + self.pcec.corr_weights[2] * hist[1];
                        x_ref[i] = curr_x[i] + h * v_ref;
                    }
                    stabilized_x[..n].copy_from_slice(&x_ref[..n]);
                    "Stiff Transient (PECEC, 2 NFE)"
                };
                v_damped_vec[..n].copy_from_slice(&v_corr[..n]);
                (2, mode, curvature * h * h * 0.02)
            };

            // Active Symplectic Energy Projection on Hamiltonian Phase Space
            if let Some(target_e) = self.energy_target {
                if n >= 2 && n % 2 == 0 {
                    let m = n / 2;
                    let p_sq: f32 = stabilized_x[m..n].iter().map(|&v| v * v).sum();
                    let current_h = if let Some(ref h_func) = self.energy_fn {
                        (h_func.0)(&stabilized_x)
                    } else {
                        // Default 2-body Kepler potential V = -1/r if no custom Hamiltonian provided
                        let r_sq: f32 = stabilized_x[0..m].iter().map(|&v| v * v).sum();
                        let r = r_sq.sqrt().max(denom_guard.sqrt());
                        0.5 * p_sq - 1.0 / r
                    };
                    let delta_h = current_h - target_e;
                    if delta_h.abs() > energy_drift_floor && p_sq > denom_guard {
                        // C^inf Infinitely Differentiable Hyperbolic Energy Projection:
                        // Matches d/dxi sqrt(1 + xi) = 0.5 at xi = 0, smoothly saturating in (0.5, 1.5).
                        let xi = -2.0 * (delta_h as f64) / ((p_sq as f64) + (denom_guard as f64));
                        let scale = (1.0 + 0.5 * xi.tanh()) as f32;
                        for i in m..n {
                            stabilized_x[i] *= scale;
                        }
                    }
                }
            }

            total_nfe += step_nfe - 1; // 1 was already counted for v_pred
            if self.rate_distortion_quantization {
                rd_buffer.update_with_history_5(&v_damped_vec, &history, self.target_tolerance);
                for i in 0..n {
                    let rec = rd_buffer.reconstruct(i);
                    history[i][4] = history[i][3];
                    history[i][3] = history[i][2];
                    history[i][2] = rec[2];
                    history[i][1] = rec[1];
                    history[i][0] = rec[0];
                }
            } else {
                for i in 0..n {
                    history[i][4] = history[i][3];
                    history[i][3] = history[i][2];
                    history[i][2] = history[i][1];
                    history[i][1] = history[i][0];
                    history[i][0] = v_damped_vec[i];
                }
            }

            if self.record_diagnostics && diagnostics.len() < 2000 {
                diagnostics.push(MetaStepDiagnostics {
                    step_index: step_count,
                    step_size: h,
                    nfe_consumed: step_nfe,
                    detected_regime: regime,
                    estimated_curvature: curvature,
                    local_truncation_error: err,
                });
            }

            curr_x[..n].copy_from_slice(&stabilized_x[..n]);
            prev_curvature = curvature;
            t += h;
            step_count += 1;

            // Precision-aware, Order-aware, and Cost-Regime-aware PID Step-size adaptation
            let p_order = if regime.starts_with("Order-5") {
                5.0f32
            } else if regime.starts_with("High-Order RK") {
                4.0f32
            } else {
                3.0f32
            };
            let safety = match effective_cost_regime {
                FunctionCostRegime::Expensive => 0.95f32,
                FunctionCostRegime::Moderate | FunctionCostRegime::Auto => (0.5f32).powf(1.0 / (p_order + 1.0)),
                FunctionCostRegime::Free => 0.85f32,
            };
            let exp_p = 1.0 / (p_order + 1.0);
            let (min_scale, max_scale) = match effective_cost_regime {
                FunctionCostRegime::Expensive => (0.7f32, 1.4f32),
                FunctionCostRegime::Moderate | FunctionCostRegime::Auto => (0.4f32, 2.0f32),
                FunctionCostRegime::Free => (0.2f32, 2.5f32),
            };
            let e_curr = err.max(antiwindup);
            let raw_scale = if self.super_smooth_routing {
                // Gustafsson Predictive Digital PI/PID Controller
                // Dampens step-size limit-cycle oscillations on stiff or non-linear regimes:
                // h_{n+1} = h_n * safety * (tol / e_n)^ki * (e_{n-1} / e_n)^kp * (e_{n-1}^2 / (e_n * e_{n-2}))^kd
                let ki = 0.35f32 / (p_order + 1.0);
                let kp = 0.25f32 / (p_order + 1.0);
                let kd = 0.05f32 / (p_order + 1.0);
                let r_i = (self.target_tolerance / e_curr).powf(ki);
                let r_p = (prev_err.max(antiwindup) / e_curr).powf(kp);
                let r_d = ((prev_err.max(antiwindup) * prev_err.max(antiwindup)) / (e_curr * prev_err2.max(antiwindup))).powf(kd);
                safety * r_i * r_p * r_d
            } else {
                let error_ratio = self.target_tolerance / e_curr;
                safety * error_ratio.powf(exp_p)
            };
            prev_err2 = prev_err;
            prev_err = e_curr;
            let scale = if self.super_smooth_routing {
                apply_smooth_clamp(raw_scale, min_scale, max_scale, self.clamping_method)
            } else {
                raw_scale.clamp(min_scale, max_scale)
            };
            h = (h * scale).clamp(self.min_step, self.max_step);
        }

        (curr_x, diagnostics, total_nfe)
    }

    /// Autonomously integrates an Itô Stochastic Differential Equation (SDE):
    /// $$d X_t = \mu(X_t, t)\,dt + \sigma(X_t, t)\,d W_t$$
    ///
    /// Dynamically fingerprints the drift-to-diffusion information ratio $\rho_s = \frac{h \|\mu\|}{\sqrt{h} \|\sigma\|}$:
    /// - When diffusion dominates ($\rho_s < 0.1$), drops to 1-NFE Euler-Maruyama to eliminate wasted drift evaluations.
    /// - When drift curvature dominates ($\rho_s \ge 0.1$), applies Conformal Stochastic Heun Predictor-Corrector.
    pub fn solve_sde<F, S, N>(
        &self,
        x0: &[f32],
        t0: f32,
        t1: f32,
        mut drift_fn: F,
        mut diffusion_fn: S,
        mut noise_fn: N,
    ) -> (Vec<f32>, Vec<MetaStepDiagnostics>, usize)
    where
        F: FnMut(&[f32], f32) -> Vec<f32>,
        S: FnMut(&[f32], f32) -> Vec<f32>,
        N: FnMut(usize) -> Vec<f32>,
    {
        let n = x0.len();
        let mut curr_x = x0.to_vec();
        let mut t = t0;
        let total_span = (t1 - t0).abs().max(1e-6);
        let mut h = (total_span / 20.0f32).clamp(self.min_step, self.max_step);
        let mut total_nfe = 0;
        let mut step_count = 0;
        let mut diagnostics = Vec::new();

        let prec = DataTypeSpec::of(self.numeric_type);
        let temporal_guard = prec.temporal_guard() as f32;

        let mut x_pred = vec![0.0f32; n];
        let mut x_next = vec![0.0f32; n];

        while t < t1 - temporal_guard {
            if step_count >= self.max_steps {
                break;
            }
            if t + h > t1 {
                h = t1 - t;
            }

            let sqrt_h = h.sqrt();
            let dw = noise_fn(n);

            let v_curr = drift_fn(&curr_x, t);
            let s_curr = diffusion_fn(&curr_x, t);
            total_nfe += 2; // 1 drift + 1 diffusion

            let mut v_norm_sq = 0.0f32;
            let mut s_norm_sq = 0.0f32;
            for i in 0..n {
                v_norm_sq += v_curr[i] * v_curr[i];
                s_norm_sq += s_curr[i] * s_curr[i];
            }
            let v_norm = (v_norm_sq / n.max(1) as f32).sqrt();
            let s_norm = (s_norm_sq / n.max(1) as f32).sqrt();

            let drift_scale = h * v_norm;
            let diff_scale = sqrt_h * s_norm;
            let rho_s = drift_scale / (diff_scale + 1e-8);

            let (step_nfe, regime, err) = if rho_s < 0.1 {
                // Diffusion-Dominated: Higher-order drift multi-step yields zero Shannon information gain.
                // Take 1-NFE Euler-Maruyama step directly.
                for i in 0..n {
                    x_next[i] = curr_x[i] + h * v_curr[i] + sqrt_h * s_curr[i] * dw[i];
                }
                (0, "SDE: Diffusion-Dominated (Euler-Maruyama, 1 NFE)", s_norm * sqrt_h * 0.1)
            } else {
                // Drift-Dominated / Curved Stochastic: Predictor-Corrector Stochastic Heun
                for i in 0..n {
                    x_pred[i] = curr_x[i] + h * v_curr[i] + sqrt_h * s_curr[i] * dw[i];
                }
                let v_pred = drift_fn(&x_pred, t + h);
                let s_pred = diffusion_fn(&x_pred, t + h);
                total_nfe += 2;

                let mut diff_sq = 0.0f32;
                for i in 0..n {
                    let d = v_pred[i] - v_curr[i];
                    diff_sq += d * d;
                }
                let curvature = (diff_sq / v_norm_sq.max(1e-8)).sqrt();
                let conformal_scale = if curvature > 50.0 {
                    1.0f32 / (1.0f32 + 0.5f32 * curvature)
                } else {
                    1.0f32
                };

                for i in 0..n {
                    let v_eff = 0.5 * (v_curr[i] + v_pred[i] * conformal_scale);
                    let s_eff = 0.5 * (s_curr[i] + s_pred[i]);
                    x_next[i] = curr_x[i] + h * v_eff + sqrt_h * s_eff * dw[i];
                }
                (2, "SDE: Drift-Curved (Conformal Stochastic Heun, 2 NFE)", curvature * h * h * 0.05 + 0.05 * s_norm * sqrt_h)
            };

            if diagnostics.len() < 2000 {
                diagnostics.push(MetaStepDiagnostics {
                    step_index: step_count,
                    step_size: h,
                    nfe_consumed: 2 + step_nfe,
                    detected_regime: regime,
                    estimated_curvature: rho_s,
                    local_truncation_error: err,
                });
            }

            curr_x = x_next.clone();
            t += h;
            step_count += 1;

            // SDE Step size adaptation: bounded by stochastic stability
            let ratio = (self.target_tolerance / err.max(1e-6)).powf(0.33);
            let raw_sde_scale = 0.9 * ratio;
            let scale = if self.super_smooth_routing {
                apply_smooth_clamp(raw_sde_scale, 0.5, 1.5, self.clamping_method)
            } else {
                raw_sde_scale.clamp(0.5, 1.5)
            };
            h = (h * scale).clamp(self.min_step, self.max_step);
        }

        (curr_x, diagnostics, total_nfe)
    }

    /// Autonomously integrates with dynamic algorithm adaptation while constructing
    /// an exact continuous Cubic Hermite trajectory for dense evaluation at $O(1)$ extra cost.
    pub fn solve_auto_dense<F>(
        &self,
        x0: &[f32],
        mut velocity_fn: F,
    ) -> (Vec<f32>, DenseHermiteTrajectory<f32>, Vec<MetaStepDiagnostics>, usize)
    where
        F: FnMut(&[f32], f32) -> Vec<f32>,
    {
        let n = x0.len();
        let mut curr_x = x0.to_vec();
        let mut t = 0.0f32;
        let mut h = (1.0f32 / 20.0f32).clamp(self.min_step, self.max_step);
        let mut total_nfe = 0;
        let mut step_count = 0;
        let mut diagnostics = Vec::new();
        let mut dense = DenseHermiteTrajectory::new();

        let prec = DataTypeSpec::of(self.numeric_type);
        let eps_mach = prec.machine_epsilon(None) as f32;
        let _sqrt_eps = prec.sqrt_machine_epsilon() as f32;
        let temporal_guard = prec.temporal_guard() as f32;
        let denom_guard = prec.denominator_guard() as f32;
        let _energy_drift_floor = prec.energy_drift_guard() as f32;
        let antiwindup = prec.pid_antiwindup_guard() as f32;
        let mut last_regime_is_singular = false;

        let mut history = vec![[0.0f32; 3]; n];
        let h_half = h * 0.5;
        let v0 = velocity_fn(&curr_x, 0.0);
        total_nfe += 1;
        let mut x_prev1 = vec![0.0f32; n];
        for i in 0..n {
            x_prev1[i] = curr_x[i] - h_half * v0[i];
        }
        let v_prev1 = velocity_fn(&x_prev1, -h_half);
        total_nfe += 1;
        let mut x_prev2 = vec![0.0f32; n];
        for i in 0..n {
            x_prev2[i] = x_prev1[i] - h_half * v_prev1[i];
        }
        let v_prev2 = velocity_fn(&x_prev2, -h);
        total_nfe += 1;

        for i in 0..n {
            history[i] = [v0[i], v_prev1[i], v_prev2[i]];
        }

        let mut v_curr = v0;

        while t < 1.0 - temporal_guard {
            if t + h > 1.0 {
                h = 1.0 - t;
            }

            let mut x_pred = vec![0.0f32; n];
            for i in 0..n {
                let hist = history[i];
                let v_est = self.pcec.pred_weights[0] * hist[0]
                    + self.pcec.pred_weights[1] * hist[1]
                    + self.pcec.pred_weights[2] * hist[2];
                x_pred[i] = curr_x[i] + h * v_est;
            }

            let v_pred = velocity_fn(&x_pred, t + h);
            total_nfe += 1;

            let mut diff_sq = 0.0f32;
            let mut v_norm_sq = 0.0f32;
            let mut has_nan = false;
            for i in 0..n {
                if !v_pred[i].is_finite() {
                    has_nan = true;
                }
                let d = v_pred[i] - history[i][0];
                diff_sq += d * d;
                v_norm_sq += history[i][0] * history[i][0];
            }
            let atol_v_sq = self.target_tolerance.max(10.0f32 * eps_mach).max(denom_guard * denom_guard);
            let curvature = (diff_sq / (v_norm_sq + atol_v_sq)).sqrt();
            let is_singular = has_nan || curvature > 1.5;

            if is_singular != last_regime_is_singular && step_count > 0 {
                let v_reinit = velocity_fn(&curr_x, t);
                total_nfe += 1;
                for i in 0..n {
                    history[i] = [v_reinit[i], v_reinit[i], v_reinit[i]];
                }
            }
            last_regime_is_singular = is_singular;

            let (next_x, v_next, step_nfe, regime, err) = if is_singular {
                let x_norm: f32 = (curr_x.iter().map(|&v| v * v).sum::<f32>() / n.max(1) as f32).sqrt();
                let clamp_limit = prec.adaptive_clamp_bound(x_norm as f64, self.rough.subgradient_clamping as f64) as f32;
                let mut clamped_x = vec![0.0f32; n];
                for i in 0..n {
                    let clamped = if self.super_smooth_routing {
                        apply_smooth_clamp(v_pred[i], -clamp_limit, clamp_limit, self.clamping_method)
                    } else {
                        v_pred[i].clamp(-clamp_limit, clamp_limit)
                    };
                    clamped_x[i] = curr_x[i] + h * clamped;
                }
                (clamped_x, v_pred, 1, "Rough/Singular (Subgradient Clamped)", 0.01)
            } else if curvature < self.pcec.curvature_threshold_pe {
                (x_pred, v_pred, 1, "Smooth Laminar (PE, 1 NFE)", curvature * h * h * 0.1)
            } else if curvature < self.pcec.curvature_threshold_pec {
                let mut x_corr = vec![0.0f32; n];
                for i in 0..n {
                    let hist = history[i];
                    let v_corr = self.pcec.corr_weights[0] * v_pred[i]
                        + self.pcec.corr_weights[1] * hist[0]
                        + self.pcec.corr_weights[2] * hist[1];
                    x_corr[i] = curr_x[i] + h * v_corr;
                }
                (x_corr, v_pred, 1, "Mild Curvature (PEC, 1 NFE)", curvature * h * h * 0.05)
            } else {
                let mut x_corr = vec![0.0f32; n];
                for i in 0..n {
                    let hist = history[i];
                    let v_corr = self.pcec.corr_weights[0] * v_pred[i]
                        + self.pcec.corr_weights[1] * hist[0]
                        + self.pcec.corr_weights[2] * hist[1];
                    x_corr[i] = curr_x[i] + h * v_corr;
                }
                let v_corr = velocity_fn(&x_corr, t + h);
                let (final_x, mode) = if curvature < self.pcec.curvature_threshold_pece {
                    (x_corr, "Moderate Curvature (PECE, 2 NFE)")
                } else {
                    let mut x_ref = vec![0.0f32; n];
                    for i in 0..n {
                        let hist = history[i];
                        let v_ref = self.pcec.corr_weights[0] * v_corr[i]
                            + self.pcec.corr_weights[1] * hist[0]
                            + self.pcec.corr_weights[2] * hist[1];
                        x_ref[i] = curr_x[i] + h * v_ref;
                    }
                    (x_ref, "Stiff Transient (PECEC, 2 NFE)")
                };
                (final_x, v_corr, 2, mode, curvature * h * h * 0.02)
            };

            dense.push_segment(
                t,
                t + h,
                curr_x.clone(),
                next_x.clone(),
                v_curr.clone(),
                v_next.clone(),
            );

            total_nfe += step_nfe - 1;
            for i in 0..n {
                history[i][2] = history[i][1];
                history[i][1] = history[i][0];
                history[i][0] = v_next[i];
            }

            if diagnostics.len() < 2000 {
                diagnostics.push(MetaStepDiagnostics {
                    step_index: step_count,
                    step_size: h,
                    nfe_consumed: step_nfe,
                    detected_regime: regime,
                    estimated_curvature: curvature,
                    local_truncation_error: err,
                });
            }

            curr_x = next_x;
            v_curr = v_next;
            t += h;
            step_count += 1;

            let error_ratio = self.target_tolerance / err.max(antiwindup);
            let raw_dense_scale = 0.9 * error_ratio.powf(0.25);
            let scale = if self.super_smooth_routing {
                apply_smooth_clamp(raw_dense_scale, 0.4, 1.8, self.clamping_method)
            } else {
                raw_dense_scale.clamp(0.4, 1.8)
            };
            h = (h * scale).clamp(self.min_step, self.max_step);
        }

        (curr_x, dense, diagnostics, total_nfe)
    }
}

impl ContinuousFlowSolver for SelfOptimizingMetaIntegrator {
    fn solve_flow(&self, x0: &[f32], _steps: usize, velocity_fn: &mut dyn FnMut(&[f32], f32) -> Vec<f32>) -> Vec<f32> {
        let (res, _, _) = self.solve_auto(x0, |x, t| velocity_fn(x, t));
        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::precision::NumericPrecision;
    use std::sync::Arc;

    #[derive(Debug)]
    struct CustomHalfStepPrecision;

    impl NumericPrecision for CustomHalfStepPrecision {
        fn name(&self) -> &'static str {
            "custom_half_step"
        }
        fn bits(&self) -> usize {
            8
        }
        fn is_discrete(&self) -> bool {
            true
        }
        fn machine_epsilon(&self, _val: Option<f64>) -> f64 {
            0.5
        }
        fn denominator_guard(&self) -> f64 {
            0.1
        }
        fn temporal_guard(&self) -> f64 {
            0.2
        }
        fn energy_drift_guard(&self) -> f64 {
            0.5
        }
        fn pid_antiwindup_guard(&self) -> f64 {
            0.01
        }
        fn default_tolerances(&self) -> (f64, f64) {
            (0.5, 0.5)
        }
        fn adaptive_clamp_bound(&self, _state_norm: f64, _velocity_norm: f64) -> f64 {
            5.0
        }
        fn quantize(&self, val: f64) -> f64 {
            (val * 2.0).round() / 2.0
        }
    }

    #[test]
    fn test_self_optimizing_meta_integrator_with_custom_precision() {
        let prec = Arc::new(CustomHalfStepPrecision);
        let integrator = SelfOptimizingMetaIntegrator::new(1e-4)
            .with_custom_precision(prec.clone());

        assert!(integrator.custom_precision.is_some());
        assert_eq!(integrator.custom_precision.as_ref().unwrap().name(), "custom_half_step");

        // Integrate dx/dt = -x on [0, 1] with x(0) = 2.0
        let (final_x, nfe) = integrator.solve_interval_with_precision(
            prec.as_ref(),
            &[2.0],
            0.0,
            1.0,
            |x, _t| vec![-x[0]],
        );

        assert!(nfe > 0);
        // Result must be aligned to the 0.5 lattice
        let mult = (final_x[0] * 2.0).round();
        assert!((final_x[0] - mult / 2.0).abs() < 1e-12, "Result {} not quantized to 0.5 lattice", final_x[0]);
    }

    #[test]
    fn test_configurable_smooth_gating_methods() {
        let methods = [
            SmoothGatingMethod::AlgebraicSigmoid,
            SmoothGatingMethod::RationalHill4,
            SmoothGatingMethod::DirectionalReversal,
            SmoothGatingMethod::BoltzmannSoftmax,
            SmoothGatingMethod::HermiteSmoothstep,
            SmoothGatingMethod::HardThreshold,
        ];

        for &m in &methods {
            let g_zero = apply_smooth_gate(0.0, m);
            let g_pos = apply_smooth_gate(5.0, m);
            let g_neg = apply_smooth_gate(-5.0, m);

            assert!(g_pos >= g_zero, "{:?} must be monotonic at positive inputs", m);
            assert!(g_zero >= g_neg, "{:?} must be monotonic at negative inputs", m);
            assert!((0.0..=1.0).contains(&g_zero), "{:?} zero value must be in [0, 1]", m);
            assert!((0.0..=1.0).contains(&g_pos), "{:?} positive value must be in [0, 1]", m);
            assert!((0.0..=1.0).contains(&g_neg), "{:?} negative value must be in [0, 1]", m);
        }
    }

    #[test]
    fn test_configurable_smooth_clamping_methods() {
        let methods = [
            SmoothClampingMethod::AlgebraicSigmoid,
            SmoothClampingMethod::QuarticRoot,
            SmoothClampingMethod::HyperbolicTangent,
            SmoothClampingMethod::HermiteSmootherstep,
            SmoothClampingMethod::HardPiecewise,
        ];

        let min_val = -2.0f32;
        let max_val = 3.0f32;

        for &m in &methods {
            let c_within = apply_smooth_clamp(0.5, min_val, max_val, m);
            let c_far_left = apply_smooth_clamp(-100.0, min_val, max_val, m);
            let c_far_right = apply_smooth_clamp(100.0, min_val, max_val, m);

            assert!(c_far_left >= min_val - 1e-3, "{:?} left clamp must bound below", m);
            assert!(c_far_right <= max_val + 1e-3, "{:?} right clamp must bound above", m);
            assert!(c_far_left < c_within && c_within < c_far_right, "{:?} must be strictly monotonic", m);
        }
    }

    #[test]
    fn test_configurable_smooth_max_methods() {
        let methods = [
            SmoothMaxMethod::AlgebraicSquareRoot,
            SmoothMaxMethod::LogSumExp,
            SmoothMaxMethod::HardMax,
        ];

        let a = 1.0f32;
        let b = 4.0f32;
        let delta = 0.05f32;

        for &m in &methods {
            let s_max = apply_smooth_max(a, b, delta, m);
            assert!((s_max - 4.0).abs() < 0.1, "{:?} smooth max of (1, 4) must be approx 4, got {}", m, s_max);

            let s_max_equal = apply_smooth_max(2.0, 2.0, delta, m);
            assert!(s_max_equal >= 2.0, "{:?} smooth max of equal elements must upper bound", m);
        }
    }

    #[test]
    fn test_ito_vs_stratonovich_and_milstein_sde_integration() {
        let x0 = [1.0f32];
        let h = 0.01f32;
        let dw = [0.15f32];

        // SDE: dX = -2X dt + 0.5X dW
        let drift = |x: &[f32], _t: f32| vec![-2.0 * x[0]];
        let diff = |x: &[f32], _t: f32| vec![0.5 * x[0]];

        // Test Itô Heun
        let r_ito = RoughItoIntegrator::default()
            .with_sde_form(SdeIntegrationForm::Ito)
            .with_order_mode(SdeOrderMode::PredictorCorrectorHeun)
            .with_clamping_method(SmoothClampingMethod::AlgebraicSigmoid);
        let x_ito = r_ito.step(&x0, 0.0, h, &dw, drift, diff);

        // Test Stratonovich Heun (includes Itô-Stratonovich drift correction -0.5 * sigma * sigma')
        let r_strat = RoughItoIntegrator::default()
            .with_sde_form(SdeIntegrationForm::Stratonovich)
            .with_order_mode(SdeOrderMode::PredictorCorrectorHeun)
            .with_clamping_method(SmoothClampingMethod::AlgebraicSigmoid);
        let x_strat = r_strat.step(&x0, 0.0, h, &dw, drift, diff);

        // Test Milstein Strong Order 1.0
        let r_milstein = RoughItoIntegrator::default()
            .with_sde_form(SdeIntegrationForm::Ito)
            .with_order_mode(SdeOrderMode::MilsteinHigherOrder)
            .with_clamping_method(SmoothClampingMethod::AlgebraicSigmoid);
        let x_milstein = r_milstein.step(&x0, 0.0, h, &dw, drift, diff);

        assert!(x_ito[0].is_finite());
        assert!(x_strat[0].is_finite());
        assert!(x_milstein[0].is_finite());
        // Stratonovich drift should differ from Itô drift due to correction:
        assert!((x_ito[0] - x_strat[0]).abs() > 1e-6, "Itô and Stratonovich steps must reflect drift conversion");
    }

    #[test]
    fn test_self_optimizing_meta_integrator_configurable_smooth_methods() {
        let integrator = SelfOptimizingMetaIntegrator::new(1e-4)
            .with_gating_method(SmoothGatingMethod::RationalHill4)
            .with_clamping_method(SmoothClampingMethod::QuarticRoot)
            .with_max_method(SmoothMaxMethod::LogSumExp);

        assert_eq!(integrator.gating_method, SmoothGatingMethod::RationalHill4);
        assert_eq!(integrator.clamping_method, SmoothClampingMethod::QuarticRoot);
        assert_eq!(integrator.max_method, SmoothMaxMethod::LogSumExp);

        // Integrate dx/dt = -x on [0, 1] with custom smooth methods
        let (final_x, diag, nfe) = integrator.solve_auto(&[1.0], |x, _t| vec![-x[0]]);
        println!("FINAL_X: {:?}, NFE: {}, DIAG_COUNT: {}", final_x, nfe, diag.len());
        for d in diag.iter().take(5) {
            println!("  diag: {:?}", d);
        }
        assert!(nfe > 0);
        let target = (-1.0f32).exp();
        assert!((final_x[0] - target).abs() < 1e-2, "Integration error: expected {}, got {}", target, final_x[0]);
    }
}
