//! Weight Initialization and muP Scaling (C03 Topic 03).
//!
//! Native Rust Candle implementation of:
//! 1. LayerScale: Residual branch modulation (Touvron et al. 2021).
//! 2. ReZero: Residual branch scalar gate initialized to zero (Zhang et al. 2019; Bachlechner et al. 2020).
//! 3. Residual 1/sqrt(2L) Scale: Depth variance stabilization.
//! 4. muP Readout: Output projection with 1/width_scale multiplier (Yang et al. 2022).

use candle_core::{Result, Tensor};

/// Linear anchor forward.
pub struct LinearAnchor {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl LinearAnchor {
    pub fn new(weight: Tensor, bias: Tensor) -> Self {
        Self { weight, bias }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = x.matmul(&self.weight)?;
        out.broadcast_add(&self.bias)
    }
}

/// LayerScale forward: x + gamma * residual
pub fn layerscale_forward(x: &Tensor, residual: &Tensor, gamma: &Tensor) -> Result<Tensor> {
    let scaled = residual.broadcast_mul(gamma)?;
    x + &scaled
}

/// ReZero forward: x + alpha * residual
pub fn rezero_forward(x: &Tensor, residual: &Tensor, alpha: &Tensor) -> Result<Tensor> {
    let scaled = residual.broadcast_mul(alpha)?;
    x + &scaled
}

/// Residual 1/sqrt(2L) scaling forward: x + scale * residual
pub fn residual_scale_forward(x: &Tensor, residual: &Tensor, scale: f64) -> Result<Tensor> {
    let scaled = (residual * scale)?;
    x + &scaled
}

/// muP Readout forward: (x @ weight) / width_scale
pub fn mup_readout_forward(x: &Tensor, weight: &Tensor, width_scale: f64) -> Result<Tensor> {
    let proj = x.matmul(weight)?;
    proj / width_scale
}
