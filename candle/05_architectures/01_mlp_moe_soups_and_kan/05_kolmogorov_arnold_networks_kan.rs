//! Category 05 Topic 05: Kolmogorov-Arnold Networks (KAN): B-Spline KAN, FastKAN (RBF), and ChebyKAN.
//!
//! Provides native Candle implementations for:
//! 1. ChebyKANLayer: Orthogonal Chebyshev polynomial evaluations on edge-activations.
//! 2. FastKANLayer: Fast Gaussian Radial Basis Functions (RBF) and SiLU base linear branch.

use candle_core::{Result, Tensor};

/// Evaluates SiLU (Swish) activation: x / (1.0 + exp(-x)).
pub fn silu(x: &Tensor) -> Result<Tensor> {
    let neg_x = x.neg()?;
    let exp_neg = neg_x.exp()?;
    let denom = exp_neg.affine(1.0, 1.0)?; // exp(-x) + 1.0
    x.div(&denom)
}

/// Evaluates Chebyshev polynomials T_0(u) .. T_degree(u) for u = tanh(x).
/// Returns a Tensor of shape (..., in_features, degree + 1).
pub fn chebyshev_polynomials(x: &Tensor, degree: usize) -> Result<Tensor> {
    let u = x.tanh()?;
    let ones = u.ones_like()?;

    let mut t_terms = Vec::with_capacity(degree + 1);
    t_terms.push(ones);
    if degree >= 1 {
        t_terms.push(u.clone());
    }

    for d in 2..=degree {
        let two_u = (u.affine(2.0, 0.0))?; // 2.0 * u
        let t_prev = &t_terms[d - 1];
        let t_prev2 = &t_terms[d - 2];
        let term1 = (two_u * t_prev)?;
        let t_curr = term1.sub(t_prev2)?;
        t_terms.push(t_curr);
    }

    Tensor::stack(&t_terms, x.rank())
}

/// ChebyKAN forward pass.
///
/// x: [B, S, D_in]
/// cheby_c: [D_in, D_out, degree + 1]
pub fn chebykan_forward(x: &Tensor, cheby_c: &Tensor, degree: usize) -> Result<Tensor> {
    let (b_sz, s_len, in_dim) = x.dims3()?;
    let (_in_c, out_dim, deg_plus_1) = cheby_c.dims3()?;

    // Compute Chebyshev basis: [B, S, D_in, degree + 1]
    let t_x = chebyshev_polynomials(x, degree)?;

    // Reshape T_x to [B * S, D_in * (degree + 1)]
    let t_flat = t_x.reshape((b_sz * s_len, in_dim * deg_plus_1))?;

    // Permute cheby_c from [D_in, D_out, degree + 1] to [D_in, degree + 1, D_out]
    let c_permuted = cheby_c.permute((0, 2, 1))?.contiguous()?;
    let c_flat = c_permuted.reshape((in_dim * deg_plus_1, out_dim))?;

    // Matrix multiply: [B * S, D_in * (deg + 1)] @ [D_in * (deg + 1), D_out] -> [B * S, D_out]
    let out_flat = t_flat.matmul(&c_flat)?;

    out_flat.reshape((b_sz, s_len, out_dim))
}

/// FastKAN forward pass using SiLU base and Gaussian Radial Basis Functions.
///
/// x: [B, S, D_in]
/// base_w: [D_in, D_out]
/// rbf_w: [D_in, num_grids, D_out]
/// grid: [num_grids]
/// denominator: f64
pub fn fastkan_forward(
    x: &Tensor,
    base_w: &Tensor,
    rbf_w: &Tensor,
    grid: &Tensor,
    denominator: f64,
) -> Result<Tensor> {
    let (b_sz, s_len, in_dim) = x.dims3()?;
    let (_in_r, num_grids, out_dim) = rbf_w.dims3()?;

    // 1. Base branch: silu(x) @ base_w
    let silu_x = silu(x)?;
    let silu_flat = silu_x.reshape((b_sz * s_len, in_dim))?;
    let base_out_flat = silu_flat.matmul(base_w)?;
    let base_out = base_out_flat.reshape((b_sz, s_len, out_dim))?;

    // 2. RBF branch: exp(-((x - grid) / denominator)^2)
    // Expand x to [B, S, D_in, 1] and broadcast subtract grid [num_grids] -> [B, S, D_in, num_grids]
    let x_unsq = x.unsqueeze(3)?;
    let diff = x_unsq.broadcast_sub(grid)?;
    let scaled_diff = diff.affine(1.0 / denominator, 0.0)?;
    let sqr_diff = scaled_diff.sqr()?;
    let neg_sqr = sqr_diff.neg()?;
    let phi = neg_sqr.exp()?;

    // Reshape phi to [B * S, D_in * num_grids]
    let phi_flat = phi.reshape((b_sz * s_len, in_dim * num_grids))?;
    // Reshape rbf_w to [D_in * num_grids, D_out]
    let rbf_w_flat = rbf_w.reshape((in_dim * num_grids, out_dim))?;

    // Contraction: [B * S, D_in * num_grids] @ [D_in * num_grids, D_out]
    let rbf_out_flat = phi_flat.matmul(&rbf_w_flat)?;
    let rbf_out = rbf_out_flat.reshape((b_sz, s_len, out_dim))?;

    base_out.add(&rbf_out)
}
