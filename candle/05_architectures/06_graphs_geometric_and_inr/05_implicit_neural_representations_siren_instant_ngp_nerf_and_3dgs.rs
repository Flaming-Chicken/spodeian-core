//! Implicit Neural Representations: SIREN, Instant-NGP, NeRF, and 3DGS
//! in candle/05_architectures/06_graphs_geometric_and_inr.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Sinusoidal Representation Network (SIREN) layer: sin(omega_0 * (x @ W + b)).
pub fn siren_layer(x: &Tensor, weight: &Tensor, bias: &Tensor, omega_0: f64) -> Result<Tensor> {
    let proj = x.matmul(weight)?.broadcast_add(bias)?;
    let scaled = proj.affine(omega_0, 0.0)?;
    scaled.sin()
}

/// NeRF volume rendering numerical integration along rays using transmittance quadrature.
pub fn volume_rendering_quadrature(
    densities: &Tensor,
    colors: &Tensor,
    deltas: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (b, s) = densities.dims2()?;
    let tau = (densities.relu()? * deltas)?;
    let alpha = tau.affine(-1.0, 0.0)?.exp()?.affine(-1.0, 1.0)?;

    let tau_vec = tau.to_vec2::<f32>()?;
    let mut prev_tau_vec = vec![0.0f32; b * s];
    for i in 0..b {
        let mut acc = 0.0f32;
        for j in 0..s {
            prev_tau_vec[i * s + j] = acc;
            acc += tau_vec[i][j];
        }
    }
    let prev_tau = Tensor::from_vec(prev_tau_vec, (b, s), densities.device())?;
    let transmittance = prev_tau.affine(-1.0, 0.0)?.exp()?;

    let weights = (transmittance * alpha)?.unsqueeze(2)?;
    let rendered_color = colors.broadcast_mul(&weights)?.sum(1)?;
    let accum_opacity = weights.sum(1)?;

    Ok((rendered_color, accum_opacity))
}

/// 3D Gaussian Splatting (3DGS) 2D projected Gaussian alpha evaluation.
pub fn gaussian_splat_2d_eval(x: &Tensor, mu: &Tensor, cov_inv: &Tensor) -> Result<Tensor> {
    let delta = x.broadcast_sub(mu)?;
    let proj = delta.matmul(cov_inv)?;
    let mahalanobis = (proj * delta)?.sum_keepdim(1)?;
    mahalanobis.affine(-0.5, 0.0)?.exp()
}
