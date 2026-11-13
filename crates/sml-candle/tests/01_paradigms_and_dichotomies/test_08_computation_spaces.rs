//! Integration test for Category 01 Topic 08: Computation Spaces (Candle Track).

use ai_candle::c01_paradigms_and_dichotomies::t08_computation_spaces_latent_reasoning_and_jepa::{
    ContinuousLatentReasoner, JEPAPredictor, PonderNetHalting,
};
use candle_core::{Device, Result, Tensor, safetensors::load};

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_08_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "01_paradigms_and_dichotomies/08_computation_spaces_latent_reasoning_and_jepa.safetensors",
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

    // 2. Continuous Latent Reasoning Parity
    let w_proj = tensors.get("layers.w_proj").expect("w_proj");
    let b_proj = tensors.get("layers.b_proj").expect("b_proj");
    let w_z = tensors.get("layers.w_z").expect("w_z");
    let w_anc = tensors.get("layers.w_anc").expect("w_anc");
    let b_z = tensors.get("layers.b_z").expect("b_z");
    let gold_trajectories = tensors
        .get("layers.latent_trajectories")
        .expect("latent_trajectories");
    let gold_final = tensors.get("layers.final_thought").expect("final_thought");

    let reasoner = ContinuousLatentReasoner::new(
        w_proj.clone(),
        b_proj.clone(),
        w_z.clone(),
        w_anc.clone(),
        b_z.clone(),
    );
    let (trajectories, final_thought) = reasoner.forward(test_in, 3)?;

    // Trajectories stack shape: [batch, k+1, dim]
    let refs: Vec<&Tensor> = trajectories.iter().collect();
    let stacked_trajectories = Tensor::stack(&refs, 1)?;
    let diff_traj = stacked_trajectories
        .sub(gold_trajectories)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_traj < 1e-3,
        "Latent trajectories parity error {:.2e} exceeds 1e-3",
        diff_traj
    );

    let diff_final = final_thought
        .sub(gold_final)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_final < 1e-3,
        "Final thought parity error {:.2e} exceeds 1e-3",
        diff_final
    );

    // 3. PonderNet Halting Hazards & Cumulative Probabilities
    let w_h = tensors.get("layers.w_h").expect("w_h");
    let b_h = tensors.get("layers.b_h").expect("b_h");
    let gold_halting = tensors.get("layers.halting_probs").expect("halting_probs");

    let ponder = PonderNetHalting::new(w_h.clone(), b_h.clone());
    let halting_probs = ponder.forward(&trajectories)?;
    let diff_halting = halting_probs
        .sub(gold_halting)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_halting < 1e-4,
        "Halting probs parity error {:.2e} exceeds 1e-4",
        diff_halting
    );

    // Invariant: Halting probability simplex constraint (sum <= 1.0, non-negative)
    let min_prob = halting_probs.flatten_all()?.min(0)?.to_scalar::<f32>()?;
    assert!(
        min_prob >= 0.0,
        "Negative halting probability found: {}",
        min_prob
    );
    let cum_prob = halting_probs.sum_keepdim(1)?;
    let max_cum = cum_prob.flatten_all()?.max(0)?.to_scalar::<f32>()?;
    assert!(
        max_cum <= 1.0 + 1e-5,
        "Cumulative halting probability exceeds 1.0: {}",
        max_cum
    );

    // 4. JEPA Representation Prediction Parity
    let w_p1 = tensors.get("layers.w_p1").expect("w_p1");
    let b_p1 = tensors.get("layers.b_p1").expect("b_p1");
    let w_p2 = tensors.get("layers.w_p2").expect("w_p2");
    let b_p2 = tensors.get("layers.b_p2").expect("b_p2");
    let gold_jepa_pred = tensors.get("layers.jepa_pred").expect("jepa_pred");
    let target_sy = tensors.get("layers.target_sy").expect("target_sy");
    let gold_jepa_err = tensors.get("layers.jepa_error").expect("jepa_error");

    let jepa = JEPAPredictor::new(w_p1.clone(), b_p1.clone(), w_p2.clone(), b_p2.clone());
    let jepa_pred = jepa.forward(&final_thought)?;
    let diff_jepa = jepa_pred
        .sub(gold_jepa_pred)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_jepa < 1e-3,
        "JEPA prediction parity error {:.2e} exceeds 1e-3",
        diff_jepa
    );

    let jepa_err = jepa.loss(&jepa_pred, target_sy)?;
    let diff_err = jepa_err
        .squeeze(1)?
        .sub(gold_jepa_err)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_err < 2e-3,
        "JEPA loss parity error {:.2e} exceeds 2e-3",
        diff_err
    );

    Ok(())
}
