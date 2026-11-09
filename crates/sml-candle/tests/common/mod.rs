use candle_core::{Result, Tensor};
use std::path::{Path, PathBuf};

/// Locates a weights file relative to either repository root or crate directory.
pub fn get_weights_path(rel: &str) -> PathBuf {
    let p1 = Path::new("weights").join(rel);
    if p1.exists() {
        return p1;
    }
    let p2 = Path::new("../../weights").join(rel);
    if p2.exists() {
        return p2;
    }
    panic!("Could not locate weights file: {}", rel);
}

/// Asserts that the max-norm difference ||a - b||_inf is below the specified tolerance.
#[allow(dead_code)]
pub fn assert_max_diff(a: &Tensor, b: &Tensor, tol: f32) -> Result<()> {
    let diff = a.sub(b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
    assert!(
        diff < tol,
        "Max absolute difference {} exceeds tolerance {}",
        diff,
        tol
    );
    Ok(())
}
