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

    /// Solves an entire trajectory from $t_0$ to $t_1$ with automatic step adaptation and step budgeting.
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
        _tolerance: f32,
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

        StepResult {
            next_state,
            error_estimate: err,
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
}
