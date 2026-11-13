//! Integration test for Category 03 Topic 12: Latent Decision Gates (Candle Track).

use sml_candle::c03_building_blocks_and_losses::t12_latent_decision_gates_highway_hard_concrete_entmax_and_ponder_halting::{
    film_forward, glu_forward, hard_concrete_forward, highway_forward, pondernet_halting_forward,
    sparsemax_forward, LinearAnchor,
};
use candle_core::{safetensors::load, Device, Result};

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_12_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "03_building_blocks_and_losses/12_latent_decision_gates_highway_hard_concrete_entmax_and_ponder_halting.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let test_in = tensors.get("test_input").expect("test_input");
    let test_out = tensors.get("test_output").expect("test_output");
    let weight = tensors.get("weight").expect("weight");
    let bias = tensors.get("bias").expect("bias");

    let highway_x = tensors.get("highway_x").expect("highway_x");
    let w_h = tensors.get("w_h").expect("w_h");
    let b_h = tensors.get("b_h").expect("b_h");
    let w_t = tensors.get("w_t").expect("w_t");
    let b_t = tensors.get("b_t").expect("b_t");
    let expected_highway = tensors.get("expected_highway").expect("expected_highway");

    let glu_x = tensors.get("glu_x").expect("glu_x");
    let w_glu = tensors.get("w_glu").expect("w_glu");
    let b_glu = tensors.get("b_glu").expect("b_glu");
    let expected_glu = tensors.get("expected_glu").expect("expected_glu");

    let log_alpha = tensors.get("log_alpha").expect("log_alpha");
    let hard_concrete_u = tensors.get("hard_concrete_u").expect("hard_concrete_u");
    let expected_hc_g = tensors
        .get("expected_hard_concrete_g")
        .expect("expected_hard_concrete_g");
    let expected_hc_l0 = tensors
        .get("expected_hard_concrete_l0")
        .expect("expected_hard_concrete_l0");

    let sparsemax_logits = tensors.get("sparsemax_logits").expect("sparsemax_logits");
    let expected_sparsemax = tensors
        .get("expected_sparsemax")
        .expect("expected_sparsemax");

    let ponder_lambdas = tensors.get("ponder_lambdas").expect("ponder_lambdas");
    let expected_ponder_p = tensors.get("expected_ponder_p").expect("expected_ponder_p");
    let expected_ponder_steps = tensors
        .get("expected_ponder_steps")
        .expect("expected_ponder_steps");
    let expected_ponder_kl = tensors
        .get("expected_ponder_kl")
        .expect("expected_ponder_kl");

    let film_x = tensors.get("film_x").expect("film_x");
    let film_gamma = tensors.get("film_gamma").expect("film_gamma");
    let film_beta = tensors.get("film_beta").expect("film_beta");
    let expected_film = tensors.get("expected_film").expect("expected_film");

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

    // 2. Highway Network Parity
    let highway_pred = highway_forward(highway_x, w_h, b_h, w_t, b_t)?;
    let diff_highway = highway_pred
        .sub(expected_highway)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_highway < 1e-5,
        "Highway parity error {:.2e} exceeds 1e-5",
        diff_highway
    );

    // 3. GLU Parity
    let glu_pred = glu_forward(glu_x, w_glu, b_glu)?;
    let diff_glu = glu_pred
        .sub(expected_glu)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_glu < 1e-5,
        "GLU parity error {:.2e} exceeds 1e-5",
        diff_glu
    );

    // 4. Hard Concrete Parity
    let (hc_g_pred, hc_l0_pred) =
        hard_concrete_forward(log_alpha, 0.67, -0.1, 1.1, hard_concrete_u)?;
    let diff_hc_g = hc_g_pred
        .sub(expected_hc_g)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_hc_l0 = hc_l0_pred
        .sub(expected_hc_l0)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_hc_g < 1e-5,
        "Hard concrete g parity error {:.2e} exceeds 1e-5",
        diff_hc_g
    );
    assert!(
        diff_hc_l0 < 1e-5,
        "Hard concrete L0 parity error {:.2e} exceeds 1e-5",
        diff_hc_l0
    );

    // 5. Sparsemax Parity
    let sparsemax_pred = sparsemax_forward(sparsemax_logits)?;
    let diff_sparsemax = sparsemax_pred
        .sub(expected_sparsemax)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_sparsemax < 1e-5,
        "Sparsemax parity error {:.2e} exceeds 1e-5",
        diff_sparsemax
    );

    // 6. PonderNet Halting Parity
    let (ponder_p_pred, ponder_steps_pred, ponder_kl_pred) =
        pondernet_halting_forward(ponder_lambdas, 0.3, 1e-8)?;
    let diff_p = ponder_p_pred
        .sub(expected_ponder_p)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_steps = ponder_steps_pred
        .sub(expected_ponder_steps)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_kl = ponder_kl_pred
        .sub(expected_ponder_kl)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_p < 1e-5,
        "PonderNet p parity error {:.2e} exceeds 1e-5",
        diff_p
    );
    assert!(
        diff_steps < 1e-5,
        "PonderNet steps parity error {:.2e} exceeds 1e-5",
        diff_steps
    );
    assert!(
        diff_kl < 1e-5,
        "PonderNet KL parity error {:.2e} exceeds 1e-5",
        diff_kl
    );

    // 7. FiLM Parity
    let film_pred = film_forward(film_x, film_gamma, film_beta)?;
    let diff_film = film_pred
        .sub(expected_film)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_film < 1e-5,
        "FiLM parity error {:.2e} exceeds 1e-5",
        diff_film
    );

    Ok(())
}
