//! Neuro-Symbolic Differentiable Logic, VSA, and Combinatorial Solvers
//! in candle/05_architectures/07_multimodal_neuro_symbolic_and_neuromorphic.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Differentiable Łukasiewicz fuzzy logic operators: (t_norm, t_conorm, implication).
pub fn lukasiewicz_logic(a: &Tensor, b: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
    let a_plus_b = a.add(b)?;
    let t_norm = (a_plus_b.affine(1.0, -1.0))?.clamp(0.0f32, 1.0f32)?;
    let t_conorm = a_plus_b.clamp(0.0f32, 1.0f32)?;
    let one_minus_a = a.affine(-1.0, 1.0)?;
    let implication = one_minus_a.add(b)?.clamp(0.0f32, 1.0f32)?;
    Ok((t_norm, t_conorm, implication))
}

fn l2_normalize(t: &Tensor, dim: usize) -> Result<Tensor> {
    let norm = t.sqr()?.sum_keepdim(dim)?.sqrt()?;
    let norm_clamped = norm.clamp(1e-12f32, f32::MAX)?;
    t.broadcast_div(&norm_clamped)
}

/// Vector Symbolic Architecture (VSA / HDC): binding, bundling, and associative similarity.
pub fn vsa_bind_and_bundle(
    key: &Tensor,
    val: &Tensor,
    other: &Tensor,
    codebook: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    let bound = key.mul(val)?;
    let unnorm_bundled = bound.add(other)?;
    let bundled = l2_normalize(&unnorm_bundled, 1)?;

    let norm_cb = l2_normalize(codebook, 1)?;
    let sim = bundled.matmul(&norm_cb.t()?)?;
    Ok((bound, bundled, sim))
}

/// Differentiable Sinkhorn-Knopp assignment solver on [B, N, N] logits.
pub fn sinkhorn_knopp_solver(logits: &Tensor, num_iters: usize, tau: f64) -> Result<Tensor> {
    let scale = 1.0 / tau;
    let mut p = (logits * scale)?.exp()?;
    for _ in 0..num_iters {
        let sum_cols = (p.sum_keepdim(2)? + 1e-8f64)?;
        p = p.broadcast_div(&sum_cols)?;
        let sum_rows = (p.sum_keepdim(1)? + 1e-8f64)?;
        p = p.broadcast_div(&sum_rows)?;
    }
    Ok(p)
}
