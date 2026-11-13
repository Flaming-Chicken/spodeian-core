//! Integration test for Category 01 Topic 12: Inductive vs. Transductive (Candle Track).

use sml_candle::c01_paradigms_and_dichotomies::t12_inductive_vs_transductive_and_online_regret_vs_batch_iid::{
    ftrl_proximal_step, inductive_linear_predictor, page_hinkley_cusum, transductive_label_spreading,
};
use candle_core::{safetensors::load, Device, Result};

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_12_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "01_paradigms_and_dichotomies/12_inductive_vs_transductive_and_online_regret_vs_batch_iid.safetensors",
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

    // 2. Transductive Label Spreading
    let laplacian_s = tensors.get("laplacian_s").expect("laplacian_s");
    let y_init = tensors.get("y_init").expect("y_init");
    let f_star_gold = tensors.get("f_star").expect("f_star");
    let f_probs_gold = tensors.get("f_probs").expect("f_probs");
    let t_preds_gold = tensors
        .get("transductive_preds")
        .expect("transductive_preds");

    let (f_star_calc, f_probs_calc, t_preds_calc) =
        transductive_label_spreading(laplacian_s, y_init, 0.8)?;

    let diff_f_star = f_star_calc
        .sub(f_star_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_f_star < 1e-5,
        "F* error {:.2e} exceeds 1e-5",
        diff_f_star
    );

    let diff_f_probs = f_probs_calc
        .sub(f_probs_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_f_probs < 1e-5,
        "F probs error {:.2e} exceeds 1e-5",
        diff_f_probs
    );

    let diff_t_preds = t_preds_calc
        .sub(t_preds_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_t_preds < 1e-5,
        "Transductive preds error {:.2e} exceeds 1e-5",
        diff_t_preds
    );

    // 3. Inductive Predictor
    let node_coords = tensors.get("node_coords").expect("node_coords");
    let x_unlabeled = node_coords.narrow(0, 4, 4)?;
    let w_ind = tensors.get("w_ind").expect("w_ind");
    let ind_scores_gold = tensors.get("inductive_scores").expect("inductive_scores");
    let ind_preds_gold = tensors.get("inductive_preds").expect("inductive_preds");

    let (ind_scores_calc, ind_preds_calc) = inductive_linear_predictor(&x_unlabeled, w_ind)?;

    let diff_ind_scores = ind_scores_calc
        .sub(ind_scores_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_ind_scores < 1e-5,
        "Inductive scores error {:.2e} exceeds 1e-5",
        diff_ind_scores
    );

    let diff_ind_preds = ind_preds_calc
        .sub(ind_preds_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_ind_preds < 1e-5,
        "Inductive preds error {:.2e} exceeds 1e-5",
        diff_ind_preds
    );

    // 4. Online FTRL-Proximal
    let grad_stream = tensors.get("grad_stream").expect("grad_stream");
    let final_w_gold = tensors.get("final_w_ftrl").expect("final_w_ftrl");
    let z_gold = tensors.get("z_accum").expect("z_accum");
    let n_gold = tensors.get("n_accum").expect("n_accum");

    let (final_w_calc, z_calc, n_calc) = ftrl_proximal_step(grad_stream, 4, 0.1, 1.0, 0.5, 0.1)?;

    let diff_w = final_w_calc
        .sub(final_w_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_w < 1e-5,
        "FTRL final w error {:.2e} exceeds 1e-5",
        diff_w
    );

    let diff_z = z_calc
        .sub(z_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_z < 1e-5, "FTRL z error {:.2e} exceeds 1e-5", diff_z);

    let diff_n = n_calc
        .sub(n_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_n < 1e-5, "FTRL n error {:.2e} exceeds 1e-5", diff_n);

    // 5. Page-Hinkley CUSUM Detector
    let stream_errors = tensors.get("stream_errors").expect("stream_errors");
    let cusum_s_gold = tensors.get("cusum_s").expect("cusum_s");
    let ph_stat_gold = tensors.get("ph_stat").expect("ph_stat");
    let drift_flags_gold = tensors.get("drift_flags").expect("drift_flags");

    let (cusum_s_calc, ph_stat_calc, drift_flags_calc) =
        page_hinkley_cusum(stream_errors, 0.05, 0.5, 0.10)?;

    let diff_s = cusum_s_calc
        .sub(cusum_s_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_s < 1e-5, "CUSUM s error {:.2e} exceeds 1e-5", diff_s);

    let diff_ph = ph_stat_calc
        .sub(ph_stat_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_ph < 1e-5, "PH stat error {:.2e} exceeds 1e-5", diff_ph);

    let diff_drift = drift_flags_calc
        .sub(drift_flags_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_drift < 1e-5,
        "Drift flags error {:.2e} exceeds 1e-5",
        diff_drift
    );

    Ok(())
}
