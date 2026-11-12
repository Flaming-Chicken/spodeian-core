//! ONNX Export Dynamo and Onnxruntime Inference in candle/17_deployment_low_level_and_capstones.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Helper for 3D activation by 2D weight matrix multiplication: [B, S, D] @ [D, H] -> [B, S, H].
fn matmul_3d_2d(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    let (b, s, d) = x.dims3()?;
    let h = w.dim(1)?;
    let x_flat = x.reshape((b * s, d))?;
    let y_flat = x_flat.matmul(w)?;
    y_flat.reshape((b, s, h))
}

/// Dynamo-compatible feedforward block supporting dynamic batch and sequence lengths.
pub fn onnx_dynamo_forward(
    x: &Tensor,
    w1: &Tensor,
    b1: &Tensor,
    w2: &Tensor,
    b2: &Tensor,
) -> Result<Tensor> {
    let proj1 = matmul_3d_2d(x, w1)?.broadcast_add(b1)?;
    let h = proj1.gelu_erf()?;
    let y = matmul_3d_2d(&h, w2)?.broadcast_add(b2)?;
    Ok(y)
}
