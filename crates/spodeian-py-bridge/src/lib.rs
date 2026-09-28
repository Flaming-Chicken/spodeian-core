//! Desktop-only PyO3 FFI bridge exposing Spodeian Rust mathematical kernels,
//! data augmentations, and financial factors directly to Python and PyTorch DataLoaders.

pub mod augmentations;
pub mod factors;

pub use augmentations::*;
pub use factors::*;

use pyo3::prelude::*;

#[pyfunction]
fn py_temporal_mask(sequence: Vec<f64>, mask_ratio: f64, mask_value: f64, seed: u64) -> PyResult<Vec<f64>> {
    Ok(apply_temporal_mask(&sequence, mask_ratio, mask_value, seed))
}

#[pyfunction]
fn py_gaussian_jitter(sequence: Vec<f64>, std_dev: f64, seed: u64) -> PyResult<Vec<f64>> {
    Ok(apply_gaussian_jitter(&sequence, std_dev, seed))
}

#[pyfunction]
fn py_rolling_zscore(sequence: Vec<f64>, window: usize) -> PyResult<Vec<f64>> {
    Ok(rolling_zscore_normalize(&sequence, window))
}

#[pyfunction]
fn py_time_warp(sequence: Vec<f64>, target_len: usize) -> PyResult<Vec<f64>> {
    Ok(time_warp_resample(&sequence, target_len))
}

#[pyfunction]
fn py_log_returns(prices: Vec<f64>) -> PyResult<Vec<f64>> {
    Ok(calculate_log_returns(&prices))
}

#[pyfunction]
fn py_rolling_volatility(returns: Vec<f64>, window: usize) -> PyResult<Vec<f64>> {
    Ok(calculate_rolling_volatility(&returns, window))
}

#[pyfunction]
fn py_ema(series: Vec<f64>, span: usize) -> PyResult<Vec<f64>> {
    Ok(calculate_ema(&series, span))
}

#[pyfunction]
fn py_vwap(prices: Vec<f64>, volumes: Vec<f64>) -> PyResult<Vec<f64>> {
    Ok(calculate_vwap(&prices, &volumes))
}

/// Spodeian PyO3 Python Module
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
    Ok(())
}
