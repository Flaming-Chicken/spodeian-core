//! Desktop-only PyO3 FFI bridge exposing Spodeian Rust mathematical kernels,
//! continuous flow ODE solvers, data augmentations, and financial factors directly to Python.
//!
//! **Stability: Experimental** — API subject to change

pub mod augmentations;
pub mod factors;

#[doc = "**Stability: Experimental** — API subject to change"]
pub use augmentations::*;
#[doc = "**Stability: Experimental** — API subject to change"]
pub use factors::*;

use pyo3::prelude::*;

#[pyfunction]
fn py_temporal_mask(sequence: Vec<f64>, mask_ratio: f64, mask_value: f64, seed: u64) -> Vec<f64> {
    apply_temporal_mask(&sequence, mask_ratio, mask_value, seed)
}

#[pyfunction]
fn py_gaussian_jitter(sequence: Vec<f64>, std_dev: f64, seed: u64) -> Vec<f64> {
    apply_gaussian_jitter(&sequence, std_dev, seed)
}

#[pyfunction]
fn py_rolling_zscore(sequence: Vec<f64>, window: usize) -> Vec<f64> {
    rolling_zscore_normalize(&sequence, window)
}

#[pyfunction]
fn py_time_warp(sequence: Vec<f64>, target_len: usize) -> Vec<f64> {
    time_warp_resample(&sequence, target_len)
}

#[pyfunction]
fn py_log_returns(prices: Vec<f64>) -> Vec<f64> {
    calculate_log_returns(&prices)
}

#[pyfunction]
fn py_rolling_volatility(returns: Vec<f64>, window: usize) -> Vec<f64> {
    calculate_rolling_volatility(&returns, window)
}

#[pyfunction]
fn py_ema(series: Vec<f64>, span: usize) -> Vec<f64> {
    calculate_ema(&series, span)
}

#[pyfunction]
fn py_vwap(prices: Vec<f64>, volumes: Vec<f64>) -> Vec<f64> {
    calculate_vwap(&prices, &volumes)
}

#[pyfunction]
fn py_rk4_step(x: Vec<f32>, t: f32, h: f32, velocity: Vec<f32>) -> Vec<f32> {
    sml_runtime::solvers::Rk4FlowSolver::step(&x, t, h, |_curr_x, _curr_t| velocity.clone())
}

#[pyfunction]
fn py_velocity_verlet_step(
    q: Vec<f32>,
    p: Vec<f32>,
    h: f32,
    force: Vec<f32>,
) -> (Vec<f32>, Vec<f32>) {
    sml_runtime::solvers::SymplecticVelocityVerlet::step(&q, &p, h, |_| force.clone())
}

#[pyfunction]
fn py_flow_matching_loss(v_pred: Vec<f32>, v_target: Vec<f32>) -> PyResult<f32> {
    use sml_runtime::candle_core::{Device, Tensor};
    let dev = Device::Cpu;
    let n = v_pred.len();
    let p = Tensor::from_vec(v_pred, (n,), &dev)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let t = Tensor::from_vec(v_target, (n,), &dev)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let loss = sml_runtime::loss::flow_matching_loss(&p, &t)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    loss.to_scalar::<f32>()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn py_cross_entropy_loss(
    logits: Vec<f32>,
    targets: Vec<u32>,
    num_classes: usize,
    smoothing: f64,
) -> PyResult<f32> {
    use sml_runtime::candle_core::{Device, Tensor};
    let dev = Device::Cpu;
    let b = targets.len();
    let l = Tensor::from_vec(logits, (b, num_classes), &dev)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let t = Tensor::from_vec(targets, (b,), &dev)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let loss =
        sml_runtime::loss::cross_entropy(&l, &t, smoothing, sml_runtime::loss::Reduction::Mean)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    loss.to_scalar::<f32>()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Stateful Python-facing wrapper for [`sml_runtime::optim::SelfOptimizingMeta`].
#[pyclass]
pub struct PySelfOptimizingMeta {
    inner: sml_runtime::optim::SelfOptimizingMeta,
}

#[pymethods]
impl PySelfOptimizingMeta {
    #[new]
    #[pyo3(signature = (lr=1e-3, weight_decay=0.0))]
    fn new(lr: f64, weight_decay: f64) -> Self {
        let cfg = sml_runtime::optim::SelfOptimizingMetaConfig {
            lr,
            weight_decay,
            ..Default::default()
        };
        Self {
            inner: sml_runtime::optim::SelfOptimizingMeta::new(Vec::new(), cfg),
        }
    }

    /// Executes one coordinate-wise `SelfOptimizingMeta` update on `params` given `grads`.
    fn step(&mut self, mut params: Vec<f32>, grads: Vec<f32>) -> PyResult<Vec<f32>> {
        if params.len() != grads.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "params len ({}) != grads len ({})",
                params.len(),
                grads.len()
            )));
        }
        self.inner
            .step_slice_in_place_f32(0, &grads, &mut params)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(params)
    }
}

/// Integrates `dx/dt = f(x, t)` on `[t0, t1]` using `SelfOptimizingMetaIntegrator` and a Python callable `f(x, t) -> list[float]`.
#[pyfunction]
#[pyo3(signature = (x0, t0, t1, velocity_fn, tol=1e-4))]
fn py_meta_integrate(
    py: Python<'_>,
    x0: Vec<f32>,
    t0: f32,
    t1: f32,
    velocity_fn: PyObject,
    tol: f32,
) -> PyResult<(Vec<f32>, usize)> {
    let solver = sml_runtime::solvers::SelfOptimizingMetaIntegrator::new(tol);
    let mut call_err: Option<PyErr> = None;
    let (xf, _diag, nfe) = solver.solve_interval(&x0, t0, t1, |state, t| {
        if call_err.is_some() {
            return vec![0.0; state.len()];
        }
        match velocity_fn.call1(py, (state.to_vec(), t)) {
            Ok(res) => match res.extract::<Vec<f32>>(py) {
                Ok(v) => v,
                Err(e) => {
                    call_err = Some(e);
                    vec![0.0; state.len()]
                }
            },
            Err(e) => {
                call_err = Some(e);
                vec![0.0; state.len()]
            }
        }
    });
    if let Some(e) = call_err {
        return Err(e);
    }
    Ok((xf, nfe))
}

/// Spodeian PyO3 Python Module
#[doc = "**Stability: Experimental** — API subject to change"]
#[pymodule]
fn spodeian_py_bridge(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySelfOptimizingMeta>()?;
    m.add_function(wrap_pyfunction!(py_temporal_mask, m)?)?;
    m.add_function(wrap_pyfunction!(py_gaussian_jitter, m)?)?;
    m.add_function(wrap_pyfunction!(py_rolling_zscore, m)?)?;
    m.add_function(wrap_pyfunction!(py_time_warp, m)?)?;
    m.add_function(wrap_pyfunction!(py_log_returns, m)?)?;
    m.add_function(wrap_pyfunction!(py_rolling_volatility, m)?)?;
    m.add_function(wrap_pyfunction!(py_ema, m)?)?;
    m.add_function(wrap_pyfunction!(py_vwap, m)?)?;
    m.add_function(wrap_pyfunction!(py_rk4_step, m)?)?;
    m.add_function(wrap_pyfunction!(py_velocity_verlet_step, m)?)?;
    m.add_function(wrap_pyfunction!(py_flow_matching_loss, m)?)?;
    m.add_function(wrap_pyfunction!(py_cross_entropy_loss, m)?)?;
    m.add_function(wrap_pyfunction!(py_meta_integrate, m)?)?;
    Ok(())
}
