//! Integration test for Category 01 Topic 13: Information Geometry, MDL & Free Energy (Candle Track).

use ai_candle::c01_paradigms_and_dichotomies::t13_information_geometry_solomonoff_mdl_and_maxent_free_energy::{
    AmariDuallyFlatGeometry, LinearAnchor, ShtarkovNMLAndPrequentialMDL, VariationalFreeEnergy,
};
use candle_core::{safetensors::load, Device, Result};

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_13_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "01_paradigms_and_dichotomies/13_information_geometry_solomonoff_mdl_and_maxent_free_energy.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let test_in = tensors
        .get("test_in")
        .or_else(|| tensors.get("x_anchor"))
        .expect("test_in");
    let test_out = tensors
        .get("test_out")
        .or_else(|| tensors.get("linear_anchor_out"))
        .expect("test_out");
    let weight = tensors.get("weight").expect("weight");
    let bias = tensors.get("bias").expect("bias");

    // 1. Linear Anchor Parity
    let anchor = LinearAnchor::new(weight.clone(), bias.clone());
    let linear_pred = anchor.forward(test_in)?;
    let diff_linear = linear_pred
        .sub(test_out)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_linear < 1e-5,
        "Linear anchor parity error {:.2e} exceeds 1e-5",
        diff_linear
    );

    // 2. Amari Dually Flat Geometry
    let theta = tensors.get("theta").expect("theta");
    let psi_gold = tensors.get("psi").expect("psi");
    let eta_gold = tensors.get("eta").expect("eta");
    let phi_gold = tensors.get("phi").expect("phi");
    let gap_gold = tensors.get("legendre_gap").expect("legendre_gap");
    let fisher_gold = tensors.get("fisher_metric").expect("fisher_metric");

    let psi_calc = AmariDuallyFlatGeometry::log_partition(theta)?;
    let diff_psi = psi_calc
        .sub(psi_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_psi < 1e-5, "psi error {:.2e} exceeds 1e-5", diff_psi);

    let eta_calc = AmariDuallyFlatGeometry::expectation_param(theta)?;
    let diff_eta = eta_calc
        .sub(eta_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_eta < 1e-5, "eta error {:.2e} exceeds 1e-5", diff_eta);

    let phi_calc = AmariDuallyFlatGeometry::dual_potential(&eta_calc)?;
    let diff_phi = phi_calc
        .sub(phi_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_phi < 1e-5, "phi error {:.2e} exceeds 1e-5", diff_phi);

    let gap_calc = AmariDuallyFlatGeometry::legendre_fenchel_gap(theta, &eta_calc)?;
    let diff_gap = gap_calc
        .sub(gap_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_gap < 1e-5,
        "Legendre gap error {:.2e} exceeds 1e-5",
        diff_gap
    );

    let fisher_calc = AmariDuallyFlatGeometry::fisher_rao_metric(&eta_calc)?;
    let diff_fisher = fisher_calc
        .sub(fisher_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_fisher < 1e-5,
        "Fisher metric error {:.2e} exceeds 1e-5",
        diff_fisher
    );

    // 3. Dual Geodesics
    let theta_1 = tensors.get("theta_1").expect("theta_1");
    let theta_2 = tensors.get("theta_2").expect("theta_2");
    let eta_1 = tensors.get("eta_1").expect("eta_1");
    let eta_2 = tensors.get("eta_2").expect("eta_2");
    let e_mid_gold = tensors.get("e_geodesic_mid").expect("e_geodesic_mid");
    let m_mid_gold = tensors
        .get("m_geodesic_mid_eta")
        .expect("m_geodesic_mid_eta");
    let kl_fwd_gold = tensors.get("kl_forward").expect("kl_forward");
    let kl_rev_gold = tensors.get("kl_reverse").expect("kl_reverse");
    let bregman_gold = tensors.get("bregman_div").expect("bregman_div");

    let e_mid_calc = AmariDuallyFlatGeometry::e_geodesic(theta_1, theta_2, 0.5)?;
    let diff_emid = e_mid_calc
        .sub(e_mid_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_emid < 1e-5,
        "e-geodesic error {:.2e} exceeds 1e-5",
        diff_emid
    );

    let m_mid_calc = AmariDuallyFlatGeometry::m_geodesic_eta(eta_1, eta_2, 0.5)?;
    let diff_mmid = m_mid_calc
        .sub(m_mid_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_mmid < 1e-5,
        "m-geodesic error {:.2e} exceeds 1e-5",
        diff_mmid
    );

    let kl_fwd_calc = AmariDuallyFlatGeometry::kl_divergence(eta_1, eta_2)?;
    let diff_kl_fwd = kl_fwd_calc
        .sub(kl_fwd_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_kl_fwd < 1e-5,
        "KL forward error {:.2e} exceeds 1e-5",
        diff_kl_fwd
    );

    let kl_rev_calc = AmariDuallyFlatGeometry::kl_divergence(eta_2, eta_1)?;
    let diff_kl_rev = kl_rev_calc
        .sub(kl_rev_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_kl_rev < 1e-5,
        "KL reverse error {:.2e} exceeds 1e-5",
        diff_kl_rev
    );

    let bregman_calc = AmariDuallyFlatGeometry::bregman_divergence(theta_1, theta_2, eta_1)?;
    let diff_bregman = bregman_calc
        .sub(bregman_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_bregman < 1e-5,
        "Bregman div error {:.2e} exceeds 1e-5",
        diff_bregman
    );

    // 4. Shtarkov NML & Prequential MDL
    let stream_bits = tensors.get("stream_bits").expect("stream_bits");
    let nml_gold = tensors.get("nml_log_comp").expect("nml_log_comp");
    let preq_gold = tensors
        .get("prequential_codelength")
        .expect("prequential_codelength");

    let nml_calc = ShtarkovNMLAndPrequentialMDL::shtarkov_log_complexity(8, &device)?;
    let diff_nml = nml_calc
        .sub(nml_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_nml < 1e-5,
        "Shtarkov NML error {:.2e} exceeds 1e-5",
        diff_nml
    );

    let preq_calc = ShtarkovNMLAndPrequentialMDL::prequential_codelength(stream_bits)?;
    let diff_preq = preq_calc
        .sub(preq_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_preq < 1e-5,
        "Prequential MDL error {:.2e} exceeds 1e-5",
        diff_preq
    );

    // 5. Variational Free Energy
    let q_mu = tensors.get("q_mu").expect("q_mu");
    let q_sigma = tensors.get("q_sigma").expect("q_sigma");
    let p_prior_mu = tensors.get("p_prior_mu").expect("p_prior_mu");
    let p_prior_sigma = tensors.get("p_prior_sigma").expect("p_prior_sigma");
    let obs_x = tensors.get("obs_x").expect("obs_x");
    let a_mat = tensors.get("a_mat").expect("a_mat");
    let kl_qp_gold = tensors.get("kl_qp").expect("kl_qp");
    let recon_gold = tensors.get("recon_loss").expect("recon_loss");
    let fe_gold = tensors.get("free_energy").expect("free_energy");

    let vfe = VariationalFreeEnergy::new(0.2);
    let (kl_qp_calc, recon_calc, fe_calc) =
        vfe.compute(q_mu, q_sigma, p_prior_mu, p_prior_sigma, obs_x, a_mat)?;

    let diff_fe_kl = kl_qp_calc
        .sub(kl_qp_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_fe_kl < 1e-5,
        "Free energy KL error {:.2e} exceeds 1e-5",
        diff_fe_kl
    );

    let diff_recon = recon_calc
        .sub(recon_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_recon < 1e-5,
        "Recon loss error {:.2e} exceeds 1e-5",
        diff_recon
    );

    let diff_fe = fe_calc
        .sub(fe_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_fe < 1e-5,
        "Free energy error {:.2e} exceeds 1e-5",
        diff_fe
    );

    Ok(())
}
