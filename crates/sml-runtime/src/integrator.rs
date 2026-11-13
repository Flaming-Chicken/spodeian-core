//! Unified continuous-time stepping, adaptive ODE integration, and neural flow solving.
//!
//! **Stability: Experimental** — API subject to change
//!
//! Provides the canonical [`ContinuousIntegrator`] trait unifying:
//! - Neural flow matching and diffusion ODE solving in RainAI
//! - Classical numerical and symplectic differential equations in algebra-engine
//! - Continuous-time stochastic simulation and portfolio dynamics in Risky-Business
//! - Learned neural flow controller step budgeting in spodeian-ml

use std::fmt::Debug;

/// Result of an adaptive continuous integration step.
#[doc = "**Stability: Experimental** — API subject to change"]
#[derive(Debug, Clone, PartialEq)]
pub struct StepResult<State, T> {
    /// Integrated state at $t + \Delta t$.
    pub next_state: State,
    /// Estimated local truncation error $L_2$ norm.
    pub error_estimate: T,
    /// Recommended next step size $\Delta t_{\text{rec}}$.
    pub recommended_dt: T,
    /// Whether the step satisfies the local error tolerance and was accepted.
    pub accepted: bool,
}

/// Generalized continuous-time state integrator trait.
#[doc = "**Stability: Experimental** — API subject to change"]
pub trait ContinuousIntegrator<T: Copy + PartialOrd + Debug> {
    /// Numerical state representation (e.g. `Vec<T>`, fixed-size array, or scalar).
    type State: Clone + Send + Sync;
    /// Derivative/velocity representation (usually matching `State`).
    type Velocity;

    /// Evaluates a single integration step from $t$ to $t + \Delta t$.
    fn step<F>(
        &self,
        current_state: &Self::State,
        t: T,
        dt: T,
        tolerance: T,
        dynamics: &mut F,
    ) -> StepResult<Self::State, T>
    where
        F: FnMut(&Self::State, T) -> Self::Velocity;

    /// Solves an entire trajectory from $t_0$ to $t_1$ with fixed steps.
    fn solve_trajectory<F>(
        &self,
        initial_state: &Self::State,
        t_span: (T, T),
        initial_dt: T,
        tolerance: T,
        max_steps: usize,
        dynamics: F,
    ) -> Result<(Self::State, usize), String>
    where
        F: FnMut(&Self::State, T) -> Self::Velocity;

    /// Solves an entire trajectory with dynamic adaptive step sizing based on local truncation error estimates.
    fn solve_adaptive_trajectory<F>(
        &self,
        initial_state: &Self::State,
        t_span: (T, T),
        initial_dt: T,
        tolerance: T,
        min_dt: T,
        max_dt: T,
        max_steps: usize,
        dynamics: F,
    ) -> Result<(Self::State, usize), String>
    where
        F: FnMut(&Self::State, T) -> Self::Velocity;
}

/// Explicit Euler First-Order Continuous Stepper.
#[derive(Debug, Clone, Copy, Default)]
pub struct EulerStepper;

impl ContinuousIntegrator<f64> for EulerStepper {
    type State = Vec<f64>;
    type Velocity = Vec<f64>;

    fn step<F>(
        &self,
        current_state: &Self::State,
        t: f64,
        dt: f64,
        _tolerance: f64,
        dynamics: &mut F,
    ) -> StepResult<Self::State, f64>
    where
        F: FnMut(&Self::State, f64) -> Self::Velocity,
    {
        let v = dynamics(current_state, t);
        let next_state: Vec<f64> = current_state
            .iter()
            .zip(v.iter())
            .map(|(&x, &dx)| x + dt * dx)
            .collect();

        StepResult {
            next_state,
            error_estimate: 0.0,
            recommended_dt: dt,
            accepted: true,
        }
    }

    fn solve_trajectory<F>(
        &self,
        initial_state: &Self::State,
        t_span: (f64, f64),
        dt: f64,
        tolerance: f64,
        max_steps: usize,
        mut dynamics: F,
    ) -> Result<(Self::State, usize), String>
    where
        F: FnMut(&Self::State, f64) -> Self::Velocity,
    {
        let (t_start, t_end) = t_span;
        let mut t = t_start;
        let mut state = initial_state.clone();
        let mut steps = 0;

        let eps = (t_end - t_start).abs() * 1e-6;
        while t + eps < t_end {
            if steps >= max_steps {
                return Err(format!("Exceeded maximum step budget of {max_steps}"));
            }
            let current_dt = dt.min(t_end - t);
            let res = self.step(&state, t, current_dt, tolerance, &mut dynamics);
            state = res.next_state;
            t += current_dt;
            steps += 1;
        }

        Ok((state, steps))
    }

    fn solve_adaptive_trajectory<F>(
        &self,
        initial_state: &Self::State,
        t_span: (f64, f64),
        initial_dt: f64,
        tolerance: f64,
        min_dt: f64,
        max_dt: f64,
        max_steps: usize,
        dynamics: F,
    ) -> Result<(Self::State, usize), String>
    where
        F: FnMut(&Self::State, f64) -> Self::Velocity,
    {
        let clamped_dt = initial_dt.clamp(min_dt, max_dt);
        self.solve_trajectory(
            initial_state,
            t_span,
            clamped_dt,
            tolerance,
            max_steps,
            dynamics,
        )
    }
}

impl ContinuousIntegrator<f32> for EulerStepper {
    type State = Vec<f32>;
    type Velocity = Vec<f32>;

    fn step<F>(
        &self,
        current_state: &Self::State,
        t: f32,
        dt: f32,
        _tolerance: f32,
        dynamics: &mut F,
    ) -> StepResult<Self::State, f32>
    where
        F: FnMut(&Self::State, f32) -> Self::Velocity,
    {
        let v = dynamics(current_state, t);
        let next_state: Vec<f32> = current_state
            .iter()
            .zip(v.iter())
            .map(|(&x, &dx)| x + dt * dx)
            .collect();

        StepResult {
            next_state,
            error_estimate: 0.0,
            recommended_dt: dt,
            accepted: true,
        }
    }

    fn solve_trajectory<F>(
        &self,
        initial_state: &Self::State,
        t_span: (f32, f32),
        dt: f32,
        tolerance: f32,
        max_steps: usize,
        mut dynamics: F,
    ) -> Result<(Self::State, usize), String>
    where
        F: FnMut(&Self::State, f32) -> Self::Velocity,
    {
        let (t_start, t_end) = t_span;
        let mut t = t_start;
        let mut state = initial_state.clone();
        let mut steps = 0;

        let eps = (t_end - t_start).abs() * 1e-6;
        while t + eps < t_end {
            if steps >= max_steps {
                return Err(format!("Exceeded maximum step budget of {max_steps}"));
            }
            let current_dt = dt.min(t_end - t);
            let res = self.step(&state, t, current_dt, tolerance, &mut dynamics);
            state = res.next_state;
            t += current_dt;
            steps += 1;
        }

        Ok((state, steps))
    }

    fn solve_adaptive_trajectory<F>(
        &self,
        initial_state: &Self::State,
        t_span: (f32, f32),
        initial_dt: f32,
        tolerance: f32,
        min_dt: f32,
        max_dt: f32,
        max_steps: usize,
        dynamics: F,
    ) -> Result<(Self::State, usize), String>
    where
        F: FnMut(&Self::State, f32) -> Self::Velocity,
    {
        let clamped_dt = initial_dt.clamp(min_dt, max_dt);
        self.solve_trajectory(
            initial_state,
            t_span,
            clamped_dt,
            tolerance,
            max_steps,
            dynamics,
        )
    }
}

/// 2nd-Order Heun / Adams-Bashforth Predictor-Corrector Stepper.
#[derive(Debug, Clone, Copy, Default)]
pub struct HeunPredictorCorrector;

impl ContinuousIntegrator<f32> for HeunPredictorCorrector {
    type State = Vec<f32>;
    type Velocity = Vec<f32>;

    fn step<F>(
        &self,
        current_state: &Self::State,
        t: f32,
        dt: f32,
        tolerance: f32,
        dynamics: &mut F,
    ) -> StepResult<Self::State, f32>
    where
        F: FnMut(&Self::State, f32) -> Self::Velocity,
    {
        let n = current_state.len();
        // Predictor: Euler forward step
        let k1 = dynamics(current_state, t);
        let mut predicted = Vec::with_capacity(n);
        for i in 0..n {
            predicted.push(current_state[i] + dt * k1[i]);
        }

        // Corrector: Trapezoidal evaluation
        let k2 = dynamics(&predicted, t + dt);
        let mut next_state = Vec::with_capacity(n);
        let mut err = 0.0f32;
        for i in 0..n {
            let val = current_state[i] + 0.5 * dt * (k1[i] + k2[i]);
            let local_err = (val - predicted[i]).abs();
            if local_err > err {
                err = local_err;
            }
            next_state.push(val);
        }

        let accepted = tolerance <= 0.0 || err <= tolerance;
        let safety = 0.9f32;
        let factor = if err > 1e-7 && tolerance > 0.0 {
            (tolerance / (err + 1e-7)).sqrt() * safety
        } else {
            2.0f32
        };
        let recommended_dt = dt * factor.clamp(0.2, 5.0);

        StepResult {
            next_state,
            error_estimate: err,
            recommended_dt,
            accepted,
        }
    }

    fn solve_trajectory<F>(
        &self,
        initial_state: &Self::State,
        t_span: (f32, f32),
        dt: f32,
        tolerance: f32,
        max_steps: usize,
        mut dynamics: F,
    ) -> Result<(Self::State, usize), String>
    where
        F: FnMut(&Self::State, f32) -> Self::Velocity,
    {
        let (t_start, t_end) = t_span;
        let mut t = t_start;
        let mut state = initial_state.clone();
        let mut steps = 0;

        let eps = (t_end - t_start).abs() * 1e-6;
        while t + eps < t_end {
            if steps >= max_steps {
                return Err(format!("Exceeded maximum step budget of {max_steps}"));
            }
            let current_dt = dt.min(t_end - t);
            let res = self.step(&state, t, current_dt, tolerance, &mut dynamics);
            state = res.next_state;
            t += current_dt;
            steps += 1;
        }

        Ok((state, steps))
    }

    fn solve_adaptive_trajectory<F>(
        &self,
        initial_state: &Self::State,
        t_span: (f32, f32),
        initial_dt: f32,
        tolerance: f32,
        min_dt: f32,
        max_dt: f32,
        max_steps: usize,
        mut dynamics: F,
    ) -> Result<(Self::State, usize), String>
    where
        F: FnMut(&Self::State, f32) -> Self::Velocity,
    {
        let (t_start, t_end) = t_span;
        let mut t = t_start;
        let mut state = initial_state.clone();
        let mut steps = 0;
        let mut dt = initial_dt.clamp(min_dt, max_dt);

        let eps = (t_end - t_start).abs() * 1e-6;
        while t + eps < t_end {
            if steps >= max_steps {
                return Err(format!(
                    "Exceeded maximum step budget of {max_steps} in adaptive trajectory"
                ));
            }
            let step_dt = dt.min(t_end - t);
            let res = self.step(&state, t, step_dt, tolerance, &mut dynamics);
            if res.accepted || step_dt <= min_dt {
                state = res.next_state;
                t += step_dt;
                steps += 1;
                dt = res.recommended_dt.clamp(min_dt, max_dt);
            } else {
                dt = res.recommended_dt.clamp(min_dt, max_dt);
                if dt >= step_dt {
                    dt = (step_dt * 0.5).max(min_dt);
                }
            }
        }

        Ok((state, steps))
    }
}

impl ContinuousIntegrator<f64> for HeunPredictorCorrector {
    type State = Vec<f64>;
    type Velocity = Vec<f64>;

    fn step<F>(
        &self,
        current_state: &Self::State,
        t: f64,
        dt: f64,
        tolerance: f64,
        dynamics: &mut F,
    ) -> StepResult<Self::State, f64>
    where
        F: FnMut(&Self::State, f64) -> Self::Velocity,
    {
        let n = current_state.len();
        // Predictor: Euler forward step
        let k1 = dynamics(current_state, t);
        let mut predicted = Vec::with_capacity(n);
        for i in 0..n {
            predicted.push(current_state[i] + dt * k1[i]);
        }

        // Corrector: Trapezoidal evaluation
        let k2 = dynamics(&predicted, t + dt);
        let mut next_state = Vec::with_capacity(n);
        let mut err = 0.0f64;
        for i in 0..n {
            let val = current_state[i] + 0.5 * dt * (k1[i] + k2[i]);
            let local_err = (val - predicted[i]).abs();
            if local_err > err {
                err = local_err;
            }
            next_state.push(val);
        }

        let accepted = tolerance <= 0.0 || err <= tolerance;
        let safety = 0.9f64;
        let factor = if err > 1e-15 && tolerance > 0.0 {
            (tolerance / (err + 1e-15)).sqrt() * safety
        } else {
            2.0f64
        };
        let recommended_dt = dt * factor.clamp(0.2, 5.0);

        StepResult {
            next_state,
            error_estimate: err,
            recommended_dt,
            accepted,
        }
    }

    fn solve_trajectory<F>(
        &self,
        initial_state: &Self::State,
        t_span: (f64, f64),
        dt: f64,
        tolerance: f64,
        max_steps: usize,
        mut dynamics: F,
    ) -> Result<(Self::State, usize), String>
    where
        F: FnMut(&Self::State, f64) -> Self::Velocity,
    {
        let (t_start, t_end) = t_span;
        let mut t = t_start;
        let mut state = initial_state.clone();
        let mut steps = 0;

        let eps = (t_end - t_start).abs() * 1e-6;
        while t + eps < t_end {
            if steps >= max_steps {
                return Err(format!("Exceeded maximum step budget of {max_steps}"));
            }
            let current_dt = dt.min(t_end - t);
            let res = self.step(&state, t, current_dt, tolerance, &mut dynamics);
            state = res.next_state;
            t += current_dt;
            steps += 1;
        }

        Ok((state, steps))
    }

    fn solve_adaptive_trajectory<F>(
        &self,
        initial_state: &Self::State,
        t_span: (f64, f64),
        initial_dt: f64,
        tolerance: f64,
        min_dt: f64,
        max_dt: f64,
        max_steps: usize,
        mut dynamics: F,
    ) -> Result<(Self::State, usize), String>
    where
        F: FnMut(&Self::State, f64) -> Self::Velocity,
    {
        let (t_start, t_end) = t_span;
        let mut t = t_start;
        let mut state = initial_state.clone();
        let mut steps = 0;
        let mut dt = initial_dt.clamp(min_dt, max_dt);

        let eps = (t_end - t_start).abs() * 1e-6;
        while t + eps < t_end {
            if steps >= max_steps {
                return Err(format!(
                    "Exceeded maximum step budget of {max_steps} in adaptive trajectory"
                ));
            }
            let step_dt = dt.min(t_end - t);
            let res = self.step(&state, t, step_dt, tolerance, &mut dynamics);
            if res.accepted || step_dt <= min_dt {
                state = res.next_state;
                t += step_dt;
                steps += 1;
                dt = res.recommended_dt.clamp(min_dt, max_dt);
            } else {
                dt = res.recommended_dt.clamp(min_dt, max_dt);
                if dt >= step_dt {
                    dt = (step_dt * 0.5).max(min_dt);
                }
            }
        }

        Ok((state, steps))
    }
}

/// Classical 4th-Order Runge-Kutta (RK4) Stepper with embedded truncation error estimation.
///
/// # Examples
///
/// ```rust
/// use spodeian_ml_utils::{Rk4Stepper, ContinuousIntegrator};
///
/// let stepper = Rk4Stepper;
/// let y0 = vec![1.0];
/// // dy/dt = -y => y(t) = e^(-t)
/// let mut decay = |y: &Vec<f64>, _t: f64| vec![-y[0]];
/// let result = stepper.step(&y0, 0.0, 0.1, 1e-3, &mut decay);
/// assert!(result.accepted);
/// assert!((result.next_state[0] - (-0.1f64).exp()).abs() < 1e-5);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct Rk4Stepper;

impl ContinuousIntegrator<f64> for Rk4Stepper {
    type State = Vec<f64>;
    type Velocity = Vec<f64>;

    fn step<F>(
        &self,
        current_state: &Self::State,
        t: f64,
        dt: f64,
        tolerance: f64,
        dynamics: &mut F,
    ) -> StepResult<Self::State, f64>
    where
        F: FnMut(&Self::State, f64) -> Self::Velocity,
    {
        let n = current_state.len();
        let half_dt = 0.5 * dt;

        // k1 = f(t, y)
        let k1 = dynamics(current_state, t);

        // y_k2 = y + 0.5 * dt * k1
        let mut y_k2 = Vec::with_capacity(n);
        for i in 0..n {
            y_k2.push(current_state[i] + half_dt * k1[i]);
        }
        let k2 = dynamics(&y_k2, t + half_dt);

        // y_k3 = y + 0.5 * dt * k2
        let mut y_k3 = Vec::with_capacity(n);
        for i in 0..n {
            y_k3.push(current_state[i] + half_dt * k2[i]);
        }
        let k3 = dynamics(&y_k3, t + half_dt);

        // y_k4 = y + dt * k3
        let mut y_k4 = Vec::with_capacity(n);
        for i in 0..n {
            y_k4.push(current_state[i] + dt * k3[i]);
        }
        let k4 = dynamics(&y_k4, t + dt);

        let sixth_dt = dt / 6.0;
        let mut next_state = Vec::with_capacity(n);
        let mut max_err = 0.0f64;

        for i in 0..n {
            let next_val =
                current_state[i] + sixth_dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
            // Compare against 2nd-order RK2 estimate: y + dt * k2
            let rk2_val = current_state[i] + dt * k2[i];
            let err = (next_val - rk2_val).abs();
            if err > max_err {
                max_err = err;
            }
            next_state.push(next_val);
        }

        let accepted = tolerance <= 0.0 || max_err <= tolerance;
        let safety = 0.9f64;
        let factor = if max_err > 1e-15 && tolerance > 0.0 {
            (tolerance / (max_err + 1e-15)).cbrt() * safety
        } else {
            2.0f64
        };
        let recommended_dt = dt * factor.clamp(0.2, 5.0);

        StepResult {
            next_state,
            error_estimate: max_err,
            recommended_dt,
            accepted,
        }
    }

    fn solve_trajectory<F>(
        &self,
        initial_state: &Self::State,
        t_span: (f64, f64),
        dt: f64,
        tolerance: f64,
        max_steps: usize,
        mut dynamics: F,
    ) -> Result<(Self::State, usize), String>
    where
        F: FnMut(&Self::State, f64) -> Self::Velocity,
    {
        let (t_start, t_end) = t_span;
        let mut t = t_start;
        let mut state = initial_state.clone();
        let mut steps = 0;

        let eps = (t_end - t_start).abs() * 1e-6;
        while t + eps < t_end {
            if steps >= max_steps {
                return Err(format!("Exceeded maximum step budget of {max_steps}"));
            }
            let current_dt = dt.min(t_end - t);
            let res = self.step(&state, t, current_dt, tolerance, &mut dynamics);
            state = res.next_state;
            t += current_dt;
            steps += 1;
        }

        Ok((state, steps))
    }

    fn solve_adaptive_trajectory<F>(
        &self,
        initial_state: &Self::State,
        t_span: (f64, f64),
        initial_dt: f64,
        tolerance: f64,
        min_dt: f64,
        max_dt: f64,
        max_steps: usize,
        mut dynamics: F,
    ) -> Result<(Self::State, usize), String>
    where
        F: FnMut(&Self::State, f64) -> Self::Velocity,
    {
        let (t_start, t_end) = t_span;
        let mut t = t_start;
        let mut state = initial_state.clone();
        let mut steps = 0;
        let mut dt = initial_dt.clamp(min_dt, max_dt);

        let eps = (t_end - t_start).abs() * 1e-6;
        while t + eps < t_end {
            if steps >= max_steps {
                return Err(format!(
                    "Exceeded maximum step budget of {max_steps} in adaptive trajectory"
                ));
            }
            let step_dt = dt.min(t_end - t);
            let res = self.step(&state, t, step_dt, tolerance, &mut dynamics);
            if res.accepted || step_dt <= min_dt {
                state = res.next_state;
                t += step_dt;
                steps += 1;
                dt = res.recommended_dt.clamp(min_dt, max_dt);
            } else {
                dt = res.recommended_dt.clamp(min_dt, max_dt);
                if dt >= step_dt {
                    dt = (step_dt * 0.5).max(min_dt);
                }
            }
        }

        Ok((state, steps))
    }
}

impl ContinuousIntegrator<f32> for Rk4Stepper {
    type State = Vec<f32>;
    type Velocity = Vec<f32>;

    fn step<F>(
        &self,
        current_state: &Self::State,
        t: f32,
        dt: f32,
        tolerance: f32,
        dynamics: &mut F,
    ) -> StepResult<Self::State, f32>
    where
        F: FnMut(&Self::State, f32) -> Self::Velocity,
    {
        let n = current_state.len();
        let half_dt = 0.5 * dt;

        let k1 = dynamics(current_state, t);

        let mut y_k2 = Vec::with_capacity(n);
        for i in 0..n {
            y_k2.push(current_state[i] + half_dt * k1[i]);
        }
        let k2 = dynamics(&y_k2, t + half_dt);

        let mut y_k3 = Vec::with_capacity(n);
        for i in 0..n {
            y_k3.push(current_state[i] + half_dt * k2[i]);
        }
        let k3 = dynamics(&y_k3, t + half_dt);

        let mut y_k4 = Vec::with_capacity(n);
        for i in 0..n {
            y_k4.push(current_state[i] + dt * k3[i]);
        }
        let k4 = dynamics(&y_k4, t + dt);

        let sixth_dt = dt / 6.0;
        let mut next_state = Vec::with_capacity(n);
        let mut max_err = 0.0f32;

        for i in 0..n {
            let next_val =
                current_state[i] + sixth_dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
            let rk2_val = current_state[i] + dt * k2[i];
            let err = (next_val - rk2_val).abs();
            if err > max_err {
                max_err = err;
            }
            next_state.push(next_val);
        }

        let accepted = tolerance <= 0.0 || max_err <= tolerance;
        let safety = 0.9f32;
        let factor = if max_err > 1e-7 && tolerance > 0.0 {
            (tolerance / (max_err + 1e-7)).cbrt() * safety
        } else {
            2.0f32
        };
        let recommended_dt = dt * factor.clamp(0.2, 5.0);

        StepResult {
            next_state,
            error_estimate: max_err,
            recommended_dt,
            accepted,
        }
    }

    fn solve_trajectory<F>(
        &self,
        initial_state: &Self::State,
        t_span: (f32, f32),
        dt: f32,
        tolerance: f32,
        max_steps: usize,
        mut dynamics: F,
    ) -> Result<(Self::State, usize), String>
    where
        F: FnMut(&Self::State, f32) -> Self::Velocity,
    {
        let (t_start, t_end) = t_span;
        let mut t = t_start;
        let mut state = initial_state.clone();
        let mut steps = 0;

        let eps = (t_end - t_start).abs() * 1e-6;
        while t + eps < t_end {
            if steps >= max_steps {
                return Err(format!("Exceeded maximum step budget of {max_steps}"));
            }
            let current_dt = dt.min(t_end - t);
            let res = self.step(&state, t, current_dt, tolerance, &mut dynamics);
            state = res.next_state;
            t += current_dt;
            steps += 1;
        }

        Ok((state, steps))
    }

    fn solve_adaptive_trajectory<F>(
        &self,
        initial_state: &Self::State,
        t_span: (f32, f32),
        initial_dt: f32,
        tolerance: f32,
        min_dt: f32,
        max_dt: f32,
        max_steps: usize,
        mut dynamics: F,
    ) -> Result<(Self::State, usize), String>
    where
        F: FnMut(&Self::State, f32) -> Self::Velocity,
    {
        let (t_start, t_end) = t_span;
        let mut t = t_start;
        let mut state = initial_state.clone();
        let mut steps = 0;
        let mut dt = initial_dt.clamp(min_dt, max_dt);

        let eps = (t_end - t_start).abs() * 1e-6;
        while t + eps < t_end {
            if steps >= max_steps {
                return Err(format!(
                    "Exceeded maximum step budget of {max_steps} in adaptive trajectory"
                ));
            }
            let step_dt = dt.min(t_end - t);
            let res = self.step(&state, t, step_dt, tolerance, &mut dynamics);
            if res.accepted || step_dt <= min_dt {
                state = res.next_state;
                t += step_dt;
                steps += 1;
                dt = res.recommended_dt.clamp(min_dt, max_dt);
            } else {
                dt = res.recommended_dt.clamp(min_dt, max_dt);
                if dt >= step_dt {
                    dt = (step_dt * 0.5).max(min_dt);
                }
            }
        }

        Ok((state, steps))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euler_decay_f64() {
        // dx/dt = -x => x(t) = x_0 * exp(-t)
        let stepper = EulerStepper;
        let x0 = vec![1.0f64];
        let dt = 0.001f64;
        let (x_final, steps) = stepper
            .solve_trajectory(&x0, (0.0, 1.0), dt, 1e-4, 2000, |x, _t| vec![-x[0]])
            .expect("Solve succeeded");

        assert_eq!(steps, 1000);
        let expected = (-1.0f64).exp();
        assert!((x_final[0] - expected).abs() < 1e-3);
    }

    #[test]
    fn test_heun_decay_f32() {
        let stepper = HeunPredictorCorrector;
        let x0 = vec![1.0f32];
        let dt = 0.01f32;
        let (x_final, steps) = stepper
            .solve_trajectory(&x0, (0.0, 1.0), dt, 1e-4, 2000, |x, _t| vec![-x[0]])
            .expect("Solve succeeded");

        assert_eq!(steps, 100);
        let expected = (-1.0f32).exp();
        assert!((x_final[0] - expected).abs() < 1e-4);
    }

    #[test]
    fn test_heun_decay_f64() {
        let stepper = HeunPredictorCorrector;
        let x0 = vec![1.0f64];
        let dt = 0.01f64;
        let (x_final, steps) = stepper
            .solve_trajectory(&x0, (0.0, 1.0), dt, 1e-6, 2000, |x, _t| vec![-x[0]])
            .expect("Solve succeeded");

        assert_eq!(steps, 100);
        let expected = (-1.0f64).exp();
        assert!((x_final[0] - expected).abs() < 1e-4);
    }

    #[test]
    fn test_rk4_decay_order4_convergence() {
        let stepper = Rk4Stepper;
        let x0 = vec![1.0f64];
        let expected = (-1.0f64).exp();

        // Step size h = 0.1
        let (x_h1, _) = stepper
            .solve_trajectory(&x0, (0.0, 1.0), 0.1f64, 1e-9, 200, |x, _t| vec![-x[0]])
            .expect("Solve h=0.1");
        let err1 = (x_h1[0] - expected).abs();

        // Step size h = 0.05
        let (x_h2, _) = stepper
            .solve_trajectory(&x0, (0.0, 1.0), 0.05f64, 1e-9, 200, |x, _t| vec![-x[0]])
            .expect("Solve h=0.05");
        let err2 = (x_h2[0] - expected).abs();

        // RK4 is 4th order: doubling resolution reduces error by ~2^4 = 16x
        assert!(err1 < 1e-6);
        assert!(err2 < 1e-7);
        let ratio = err1 / err2;
        assert!(
            ratio > 14.0 && ratio < 18.0,
            "Expected ~16x error reduction, got {ratio}"
        );
    }

    #[test]
    fn test_rk4_adaptive_trajectory() {
        let stepper = Rk4Stepper;
        let x0 = vec![1.0f64];
        let (x_final, steps) = stepper
            .solve_adaptive_trajectory(
                &x0,
                (0.0f64, 1.0f64),
                0.01f64,
                1e-5f64,
                1e-4f64,
                0.2f64,
                1000,
                |x: &Vec<f64>, _t: f64| vec![-x[0]],
            )
            .expect("Adaptive solve succeeded");

        let expected = (-1.0f64).exp();
        assert!((x_final[0] - expected).abs() < 1e-5);
        assert!(
            steps <= 35,
            "Adaptive solver should converge in few steps: got {steps}"
        );
    }
}
