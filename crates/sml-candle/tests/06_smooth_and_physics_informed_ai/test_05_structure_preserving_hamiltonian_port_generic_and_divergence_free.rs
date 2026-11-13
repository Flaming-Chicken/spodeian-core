use candle_core::{Device, Result, safetensors::load};

use ai_candle::c06_smooth_and_physics_informed_ai::t05_structure_preserving_hamiltonian_port_generic_and_divergence_free as topic;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_05_structure_preserving_hamiltonian_port_generic_and_divergence_free_parity() -> Result<()>
{
    let device = Device::Cpu;
    let w = load(
        common::get_weights_path(
            "06_smooth_and_physics_informed_ai/05_structure_preserving_hamiltonian_port_generic_and_divergence_free.safetensors",
        ),
        &device,
    )?;

    // Linear anchor
    let y = topic::linear_anchor(&w["test_input"], &w["weight"], &w["bias"])?;
    common::assert_max_diff(&y, &w["test_output"], 1e-5)?;

    // Potential and gradient
    let q = w["z_state"].narrow(1, 0, 2)?;
    let p = w["z_state"].narrow(1, 2, 2)?;
    let (v_potential, grad_v) =
        topic::compute_potential_and_grad(&q, &w["w_v1"], &w["b_v1"], &w["w_v2"], &w["b_v2"])?;
    common::assert_max_diff(&v_potential, &w["v_potential"], 1e-5)?;
    common::assert_max_diff(&grad_v, &w["grad_v"], 1e-5)?;

    // Hamiltonian & Port-Hamiltonian
    let (h_total, z_dot_hnn, z_dot_phnn) =
        topic::hamiltonian_and_phnn(&w["z_state"], &grad_v, &v_potential, 0.2)?;
    common::assert_max_diff(&h_total, &w["h_total"], 1e-5)?;
    common::assert_max_diff(&z_dot_hnn, &w["z_dot_hnn"], 1e-5)?;
    common::assert_max_diff(&z_dot_phnn, &w["z_dot_phnn"], 1e-5)?;

    // Symplectic Leapfrog
    let z_leapfrog = topic::leapfrog_step(
        &q, &p, &grad_v, &w["w_v1"], &w["b_v1"], &w["w_v2"], &w["b_v2"], 0.1,
    )?;
    common::assert_max_diff(&z_leapfrog, &w["z_leapfrog"], 1e-5)?;

    // Divergence-Free Solenoidal Field
    let (psi_stream, v_solenoidal) =
        topic::solenoidal_field(&q, &w["w_psi1"], &w["b_psi1"], &w["w_psi2"], &w["b_psi2"])?;
    common::assert_max_diff(&psi_stream, &w["psi_stream"], 1e-5)?;
    common::assert_max_diff(&v_solenoidal, &w["v_solenoidal"], 1e-5)?;

    // Thermodynamic GENERIC Dynamics
    let (z_dot_generic, de_dt, ds_dt) =
        topic::generic_dynamics(&w["z_state"], &grad_v, &p, &w["l0_skew"], &w["m0_psd"])?;
    common::assert_max_diff(&z_dot_generic, &w["z_dot_generic"], 1e-5)?;
    common::assert_max_diff(&de_dt, &w["de_dt"], 1e-5)?;
    common::assert_max_diff(&ds_dt, &w["ds_dt"], 1e-5)?;

    Ok(())
}
