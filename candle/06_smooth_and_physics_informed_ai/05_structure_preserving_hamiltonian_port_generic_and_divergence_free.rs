//! Structure Preserving Hamiltonian Port Generic and Divergence Free
//! in candle/06_smooth_and_physics_informed_ai.

use candle_core::{Result, Tensor};

/// Linear anchor for cross-track parity verification.
pub fn linear_anchor(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    x.matmul(weight)?.broadcast_add(bias)
}

/// Evaluates scalar potential V(q) and analytical spatial gradient grad_q V.
pub fn compute_potential_and_grad(
    q: &Tensor,
    w1: &Tensor,
    b1: &Tensor,
    w2: &Tensor,
    b2: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let z = q.matmul(w1)?.broadcast_add(b1)?;
    let a = z.tanh()?;
    let v = a.matmul(w2)?.broadcast_add(b2)?;

    let one = Tensor::ones_like(&a)?;
    let one_minus_a2 = (one - a.sqr()?)?;
    let temp = one_minus_a2.broadcast_mul(&w2.t()?)?;
    let grad_v = temp.matmul(&w1.t()?)?;

    Ok((v, grad_v))
}

/// Evaluates total Hamiltonian energy H(q, p) = 0.5 * ||p||^2 + V(q),
/// conservative HNN dynamics, and dissipative Port-Hamiltonian dynamics.
pub fn hamiltonian_and_phnn(
    z_state: &Tensor,
    grad_v: &Tensor,
    v_potential: &Tensor,
    gamma: f64,
) -> Result<(Tensor, Tensor, Tensor)> {
    let p = z_state.narrow(1, 2, 2)?;
    let t_kinetic = p.sqr()?.sum_keepdim(1)?.affine(0.5, 0.0)?;
    let h_total = (t_kinetic + v_potential)?;

    let z_dot_hnn = Tensor::cat(&[&p, &grad_v.neg()?], 1)?;
    let p_dissipated = (grad_v.neg()? - p.affine(gamma, 0.0)?)?;
    let z_dot_phnn = Tensor::cat(&[&p, &p_dissipated], 1)?;

    Ok((h_total, z_dot_hnn, z_dot_phnn))
}

/// One-step Symplectic Leapfrog (Verlet) integration.
pub fn leapfrog_step(
    q: &Tensor,
    p: &Tensor,
    grad_v: &Tensor,
    w1: &Tensor,
    b1: &Tensor,
    w2: &Tensor,
    b2: &Tensor,
    h: f64,
) -> Result<Tensor> {
    let half_h = 0.5 * h;
    let p_half = (p - grad_v.affine(half_h, 0.0)?)?;
    let q_next = (q + p_half.affine(h, 0.0)?)?;

    let (_, grad_v_next) = compute_potential_and_grad(&q_next, w1, b1, w2, b2)?;
    let p_next = (&p_half - grad_v_next.affine(half_h, 0.0)?)?;

    Tensor::cat(&[&q_next, &p_next], 1)
}

/// Computes scalar stream function psi(x, y) and divergence-free solenoidal velocity field.
pub fn solenoidal_field(
    xy: &Tensor,
    w_psi1: &Tensor,
    b_psi1: &Tensor,
    w_psi2: &Tensor,
    b_psi2: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (psi, grad_psi) = compute_potential_and_grad(xy, w_psi1, b_psi1, w_psi2, b_psi2)?;
    let dpsi_dx = grad_psi.narrow(1, 0, 1)?;
    let dpsi_dy = grad_psi.narrow(1, 1, 1)?;
    let v_solenoidal = Tensor::cat(&[&dpsi_dy, &dpsi_dx.neg()?], 1)?;

    Ok((psi, v_solenoidal))
}

/// Evaluates thermodynamic GENERIC dynamics with orthogonal projection guarantees.
pub fn generic_dynamics(
    z_state: &Tensor,
    grad_v: &Tensor,
    p: &Tensor,
    l0_skew: &Tensor,
    m0_psd: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    let b = z_state.dim(0)?;
    let grad_e = Tensor::cat(&[grad_v, p], 1)?;
    let grad_s = z_state.neg()?;

    let mut z_dot_rows = Vec::with_capacity(b);
    let eye = Tensor::eye(4, candle_core::DType::F32, z_state.device())?;

    for i in 0..b {
        let ge_i = grad_e.narrow(0, i, 1)?.t()?; // [4, 1]
        let gs_i = grad_s.narrow(0, i, 1)?.t()?; // [4, 1]

        let norm_s_sq = (gs_i.sqr()?.sum_all()?.to_scalar::<f32>()? as f64) + 1e-8;
        let proj_s_mat = gs_i.matmul(&gs_i.t()?)?.affine(1.0 / norm_s_sq, 0.0)?;
        let p_s = (&eye - proj_s_mat)?;
        let l_i = p_s.matmul(l0_skew)?.matmul(&p_s)?;

        let norm_e_sq = (ge_i.sqr()?.sum_all()?.to_scalar::<f32>()? as f64) + 1e-8;
        let proj_e_mat = ge_i.matmul(&ge_i.t()?)?.affine(1.0 / norm_e_sq, 0.0)?;
        let p_e = (&eye - proj_e_mat)?;
        let m_i = p_e.matmul(m0_psd)?.matmul(&p_e)?;

        let z_dot_i = (l_i.matmul(&ge_i)? + m_i.matmul(&gs_i)?)?.t()?; // [1, 4]
        z_dot_rows.push(z_dot_i);
    }

    let z_dot_refs: Vec<&Tensor> = z_dot_rows.iter().collect();
    let z_dot_generic = Tensor::cat(&z_dot_refs, 0)?;

    let de_dt = grad_e.mul(&z_dot_generic)?.sum_keepdim(1)?.reshape(&[b])?;
    let ds_dt = grad_s.mul(&z_dot_generic)?.sum_keepdim(1)?.reshape(&[b])?;

    Ok((z_dot_generic, de_dt, ds_dt))
}
