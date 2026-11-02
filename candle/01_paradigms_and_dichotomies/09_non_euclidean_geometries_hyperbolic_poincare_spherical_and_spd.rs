//! Category 01 Topic 09: Non-Euclidean Geometries: Hyperbolic Poincaré, Spherical, and SPD Manifolds.
//!
//! Provides native Candle implementations for:
//! 1. Hyperbolic Poincaré Ball ($\mathbb{D}_c^d$):
//!    - Exponential and logarithmic map at origin
//!    - Möbius addition: $\mathbf{x} \oplus_c \mathbf{y}$
//!    - Geodesic distance: $d_{\mathbb{D}}^c(\mathbf{x}, \mathbf{y}) = \frac{2}{\sqrt{c}}\text{artanh}(\sqrt{c}\|(-\mathbf{x})\oplus_c \mathbf{y}\|)$
//! 2. Hyperspherical Manifold ($\mathbb{S}^{d-1}$):
//!    - Angular distance: $d_{\mathbb{S}}(\mathbf{p}, \mathbf{q}) = \arccos(\langle \mathbf{p}, \mathbf{q} \rangle)$
//!    - Contrastive alignment: $\|\mathbf{p} - \mathbf{q}\|_2^2$
//! 3. Symmetric Positive Definite Manifold ($\mathcal{S}_{++}^2$):
//!    - Closed-form spectral matrix logarithm
//!    - Log-Euclidean distance: $\|\log(\mathbf{P}) - \log(\mathbf{Q})\|_F$
//!
//! # Mathematical Invariants
//! - Boundary Constraint: $\|\mathbf{x}\| < \frac{1}{\sqrt{c}}$.
//! - Self-Distance Nullity: $d(\mathbf{x}, \mathbf{x}) = 0$.
//! - Symmetry: $d(\mathbf{x}, \mathbf{y}) = d(\mathbf{y}, \mathbf{x})$.
//!
//! ```rust
//! use candle_core::{Device, Tensor};
//! use ai_candle::c01_paradigms_and_dichotomies::t09_non_euclidean_geometries_hyperbolic_poincare_spherical_and_spd::spherical_alignment;
//!
//! let device = Device::Cpu;
//! let p = Tensor::new(&[[1.0f32, 0.0], [0.0, 1.0]], &device).unwrap();
//! let q = Tensor::new(&[[1.0f32, 0.0], [0.0, 1.0]], &device).unwrap();
//! let loss = spherical_alignment(&p, &q).unwrap();
//! assert_eq!(loss.to_vec1::<f32>().unwrap(), vec![0.0, 0.0]);
//! ```

use candle_core::{Result, Tensor};

/// Tensor-level atanh(u) = 0.5 * ln((1 + u) / (1 - u)).
pub fn tensor_atanh(u: &Tensor) -> Result<Tensor> {
    let one = Tensor::ones_like(u)?;
    let num = one.add(u)?;
    let denom = one.sub(u)?;
    let ratio = num.div(&denom)?;
    let log_ratio = ratio.log()?;
    log_ratio.affine(0.5, 0.0)
}

/// Exponential map from tangent space at origin to the Poincaré ball: $\tanh(\sqrt{c}\|v\|/2) \frac{v}{\sqrt{c}\|v\|}$.
pub fn poincare_exp_zero(v: &Tensor, c: f64) -> Result<Tensor> {
    let sqrt_c = c.sqrt();
    let v_norm = v.sqr()?.sum_keepdim(1)?.sqrt()?.affine(1.0, 1e-8)?;
    let half_v_norm = v_norm.affine(sqrt_c * 0.5, 0.0)?;
    let tanh_term = half_v_norm.tanh()?;
    let scale = tanh_term
        .affine(1.0 / sqrt_c, 0.0)?
        .broadcast_div(&v_norm)?;
    v.broadcast_mul(&scale)
}

/// Logarithmic map from Poincaré ball to tangent space at origin: $2 \text{artanh}(\sqrt{c}\|x\|) \frac{x}{\sqrt{c}\|x\|}$.
pub fn poincare_log_zero(x: &Tensor, c: f64) -> Result<Tensor> {
    let sqrt_c = c.sqrt();
    let x_norm = x.sqr()?.sum_keepdim(1)?.sqrt()?.affine(1.0, 1e-8)?;
    let scaled_norm = x_norm.affine(sqrt_c, 0.0)?;
    let atanh_term = tensor_atanh(&scaled_norm)?;
    let scale = atanh_term
        .affine(2.0 / sqrt_c, 0.0)?
        .broadcast_div(&x_norm)?;
    x.broadcast_mul(&scale)
}

/// Möbius addition $\mathbf{x} \oplus_c \mathbf{y}$ in the Poincaré ball.
pub fn poincare_mobius_add(x: &Tensor, y: &Tensor, c: f64) -> Result<Tensor> {
    let xy_inner = x.mul(y)?.sum_keepdim(1)?;
    let x_sq = x.sqr()?.sum_keepdim(1)?;
    let y_sq = y.sqr()?.sum_keepdim(1)?;

    let one = Tensor::ones_like(&xy_inner)?;
    let two_c_xy = xy_inner.affine(2.0 * c, 0.0)?;
    let c_y_sq = y_sq.affine(c, 0.0)?;
    let c_x_sq = x_sq.affine(c, 0.0)?;

    let coef_x = one.add(&two_c_xy)?.add(&c_y_sq)?;
    let coef_y = one.sub(&c_x_sq)?;

    let num = x.broadcast_mul(&coef_x)?.add(&y.broadcast_mul(&coef_y)?)?;
    let c2_x_y = x_sq.mul(&y_sq)?.affine(c * c, 0.0)?;
    let denom = one.add(&two_c_xy)?.add(&c2_x_y)?.affine(1.0, 1e-15)?;

    num.broadcast_div(&denom)
}

/// Geodesic distance between x and y in the Poincaré ball.
pub fn poincare_distance(x: &Tensor, y: &Tensor, c: f64) -> Result<Tensor> {
    let sqrt_c = c.sqrt();
    let neg_x = x.affine(-1.0, 0.0)?;
    let diff = poincare_mobius_add(&neg_x, y, c)?;
    let diff_norm = diff.sqr()?.sum_keepdim(1)?.sqrt()?;
    let scaled_diff = diff_norm.affine(sqrt_c, 0.0)?;
    let atanh_term = tensor_atanh(&scaled_diff)?;
    let dist = atanh_term.affine(2.0 / sqrt_c, 0.0)?;
    dist.squeeze(1)
}

/// Geodesic angular distance on the unit hypersphere: $\arccos(\langle \mathbf{p}, \mathbf{q} \rangle)$.
pub fn spherical_distance(p: &Tensor, q: &Tensor) -> Result<Tensor> {
    let dot = p.mul(q)?.sum_keepdim(1)?.squeeze(1)?;
    let dot_vec = dot.to_vec1::<f32>()?;
    let mut dist_vec = Vec::with_capacity(dot_vec.len());
    for d in dot_vec {
        let clamped = d.clamp(-1.0 + 1e-7, 1.0 - 1e-7);
        dist_vec.push(clamped.acos());
    }
    Tensor::new(dist_vec.as_slice(), p.device())
}

/// Contrastive alignment loss on the sphere: $\|\mathbf{p} - \mathbf{q}\|_2^2$.
pub fn spherical_alignment(p: &Tensor, q: &Tensor) -> Result<Tensor> {
    p.sub(q)?.sqr()?.sum_keepdim(1)?.squeeze(1)
}

/// Closed-form matrix logarithm for a batch of $2 \times 2$ symmetric positive-definite matrices.
pub fn spd_matrix_log_2x2(m: &Tensor) -> Result<Tensor> {
    let b = m.dim(0)?;
    let m_vec = m.to_vec3::<f32>()?;
    let mut log_m = vec![vec![vec![0.0f32; 2]; 2]; b];

    for i in 0..b {
        let a = m_vec[i][0][0];
        let off = m_vec[i][0][1];
        let d = m_vec[i][1][1];

        let tr = a + d;
        let delta = ((a - d) * (a - d) + 4.0 * off * off).sqrt().max(1e-8);
        let l1 = ((tr + delta) * 0.5).max(1e-6);
        let l2 = ((tr - delta) * 0.5).max(1e-6);

        let v1_x = l1 - d;
        let v1_y = off;
        let norm1 = (v1_x * v1_x + v1_y * v1_y).sqrt().max(1e-8);
        let u1_x = v1_x / norm1;
        let u1_y = v1_y / norm1;

        let u2_x = -u1_y;
        let u2_y = u1_x;

        let log_l1 = l1.ln();
        let log_l2 = l2.ln();

        let m00 = log_l1 * u1_x * u1_x + log_l2 * u2_x * u2_x;
        let m01 = log_l1 * u1_x * u1_y + log_l2 * u2_x * u2_y;
        let m11 = log_l1 * u1_y * u1_y + log_l2 * u2_y * u2_y;

        log_m[i][0][0] = m00;
        log_m[i][0][1] = m01;
        log_m[i][1][0] = m01;
        log_m[i][1][1] = m11;
    }

    Tensor::new(log_m, m.device())
}

/// Log-Euclidean distance between batches of $2 \times 2$ SPD matrices: $\|\log(\mathbf{P}) - \log(\mathbf{Q})\|_F$.
pub fn spd_log_euclidean_distance_2x2(p: &Tensor, q: &Tensor) -> Result<Tensor> {
    let log_p = spd_matrix_log_2x2(p)?;
    let log_q = spd_matrix_log_2x2(q)?;
    let diff = log_p.sub(&log_q)?;
    let dist_sq = diff
        .sqr()?
        .sum_keepdim(2)?
        .sum_keepdim(1)?
        .squeeze(2)?
        .squeeze(1)?;
    dist_sq.sqrt()
}

/// Non-Euclidean Module for canonical linear projection and cross-track reference anchor.
#[derive(Debug, Clone)]
pub struct NonEuclideanModel {
    pub weight: Tensor,
    pub bias: Tensor,
    pub c: f64,
}

impl NonEuclideanModel {
    pub fn new(weight: Tensor, bias: Tensor, c: f64) -> Self {
        Self { weight, bias, c }
    }

    /// Canonical linear forward pass: $\mathbf{x} \mathbf{W} + \mathbf{b}$.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.matmul(&self.weight)?.broadcast_add(&self.bias)
    }
}
