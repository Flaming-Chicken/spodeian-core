//! Integration test for Category 01 Topic 06: AI by Purpose (Candle Track).

use ai_candle::c01_paradigms_and_dichotomies::t06_ai_by_purpose_prediction_prescription_control_combinatorial_symbolic::{
    CausalUpliftEstimator, CostPredictor, SPOPlusEstimator, SymbolicEquationLearner,
};
use candle_core::{safetensors::load, Device, Result};

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_06_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "01_paradigms_and_dichotomies/06_ai_by_purpose_prediction_prescription_control_combinatorial_symbolic.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let test_in = tensors.get("test_input").expect("test_input");
    let test_out = tensors.get("test_output").expect("test_output");
    let weight = tensors.get("layers.weight").expect("layers.weight");
    let bias = tensors.get("layers.bias").expect("layers.bias");

    // 1. Canonical Linear Anchor Parity
    let linear_pred = test_in.matmul(weight)?.broadcast_add(bias)?;
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

    // 2. Cost Predictor Parity
    let w_cost1 = tensors.get("layers.w_cost1").expect("w_cost1");
    let b_cost1 = tensors.get("layers.b_cost1").expect("b_cost1");
    let w_cost2 = tensors.get("layers.w_cost2").expect("w_cost2");
    let b_cost2 = tensors.get("layers.b_cost2").expect("b_cost2");
    let c_pred_gold = tensors.get("layers.c_pred").expect("c_pred");

    let cost_model = CostPredictor::new(
        w_cost1.clone(),
        b_cost1.clone(),
        w_cost2.clone(),
        b_cost2.clone(),
    );
    let c_pred_calc = cost_model.forward(test_in)?;
    let diff_c_pred = c_pred_calc
        .sub(c_pred_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_c_pred < 1e-5,
        "CostPredictor forward parity error {:.2e} exceeds 1e-5",
        diff_c_pred
    );

    // 3. Downstream Combinatorial & SPO+ Loss Parity
    let c_star = tensors.get("layers.c_star").expect("c_star");
    let regret_gold = tensors
        .get("layers.decision_regret")
        .expect("decision_regret");
    let spo_plus_gold = tensors.get("layers.spo_plus_loss").expect("spo_plus_loss");

    let regret_calc = SPOPlusEstimator::decision_regret(&c_pred_calc, c_star)?;
    let diff_regret = regret_calc
        .sub(regret_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_regret < 1e-5,
        "Decision regret parity error {:.2e} exceeds 1e-5",
        diff_regret
    );

    let spo_plus_calc = SPOPlusEstimator::spo_plus_loss(&c_pred_calc, c_star)?;
    let diff_spo_plus = spo_plus_calc
        .sub(spo_plus_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_spo_plus < 1e-5,
        "SPO+ loss parity error {:.2e} exceeds 1e-5",
        diff_spo_plus
    );

    // Invariant: Regret >= 0 and SPO+ >= Regret
    let min_regret = regret_calc.min(0)?.to_scalar::<f32>()?;
    assert!(min_regret >= -1e-6, "Regret must be non-negative");

    // 4. Causal Uplift Estimator Parity
    let w_tau = tensors.get("layers.w_tau").expect("w_tau");
    let b_tau = tensors.get("layers.b_tau").expect("b_tau");
    let w_mu0 = tensors.get("layers.w_mu0").expect("w_mu0");
    let b_mu0 = tensors.get("layers.b_mu0").expect("b_mu0");
    let treatment_t = tensors.get("layers.treatment_t").expect("treatment_t");
    let tau_gold = tensors.get("layers.tau_pred").expect("tau_pred");
    let y_factual_gold = tensors.get("layers.y_factual").expect("y_factual");

    let causal_model =
        CausalUpliftEstimator::new(w_tau.clone(), b_tau.clone(), w_mu0.clone(), b_mu0.clone());
    let tau_calc = causal_model.predict_treatment_effect(test_in)?;
    let diff_tau = tau_calc
        .sub(tau_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_tau < 1e-5,
        "Causal uplift tau parity error {:.2e} exceeds 1e-5",
        diff_tau
    );

    let y_factual_calc = causal_model.forward(test_in, treatment_t)?;
    let diff_factual = y_factual_calc
        .sub(y_factual_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_factual < 1e-5,
        "Causal factual outcome parity error {:.2e} exceeds 1e-5",
        diff_factual
    );

    // 5. Symbolic Equation Learner (EQL) Parity
    let w_eql = tensors.get("layers.w_eql").expect("w_eql");
    let b_eql = tensors.get("layers.b_eql").expect("b_eql");
    let y_sym_gold = tensors.get("layers.y_symbolic").expect("y_symbolic");

    let eql_model = SymbolicEquationLearner::new(w_eql.clone(), b_eql.clone());
    let y_sym_calc = eql_model.forward(test_in)?;
    let diff_sym = y_sym_calc
        .sub(y_sym_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_sym < 1e-5,
        "Symbolic EQL forward parity error {:.2e} exceeds 1e-5",
        diff_sym
    );

    Ok(())
}
