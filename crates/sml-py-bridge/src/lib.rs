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
    let p = Tensor::from_vec(v_pred, (n,), &dev).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let t = Tensor::from_vec(v_target, (n,), &dev).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let loss = sml_runtime::loss::flow_matching_loss(&p, &t).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    loss.to_scalar::<f32>().map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
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
    let l = Tensor::from_vec(logits, (b, num_classes), &dev).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let t = Tensor::from_vec(targets, (b,), &dev).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let loss = sml_runtime::loss::cross_entropy(&l, &t, smoothing, sml_runtime::loss::Reduction::Mean)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    loss.to_scalar::<f32>().map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Spodeian PyO3 Python Module
#[doc = "**Stability: Experimental** — API subject to change"]
#[pymodule]
fn spodeian_py_bridge(m: &Bound<'_, PyModule>) -> PyResult<()> {
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
    Ok(())
}
