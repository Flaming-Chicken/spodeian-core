//! Integration test for Category 01 Topic 11: Statistical Inference (Candle Track).

use ai_candle::c01_paradigms_and_dichotomies::t11_statistical_inference_bayesian_frequentist_likelihoodist_mdl_and_pac_bayes::{
    anytime_valid_martingale, bayesian_conjugate_laplace, frequentist_fisher_cramer_rao,
    james_stein_estimator, pac_bayes_bound,
};
use candle_core::{safetensors::load, Device, Result};

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_11_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "01_paradigms_and_dichotomies/11_statistical_inference_bayesian_frequentist_likelihoodist_mdl_and_pac_bayes.safetensors",
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

    // 2. Frequentist Fisher & Cramér-Rao
    let x_data = tensors.get("x_data").expect("x_data");
    let theta_mle_gold = tensors.get("theta_mle").expect("theta_mle");
    let fisher_gold = tensors.get("fisher_diag").expect("fisher_diag");
    let crlb_gold = tensors
        .get("cramer_rao_lower_bound")
        .expect("cramer_rao_lower_bound");
    let lrt_gold = tensors.get("lrt_stat").expect("lrt_stat");

    let theta_0 = theta_mle_gold.zeros_like()?;
    let (mle_calc, fisher_calc, crlb_calc, lrt_calc) =
        frequentist_fisher_cramer_rao(x_data, &theta_0, 0.5)?;

    let diff_mle = mle_calc
        .sub(theta_mle_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_mle < 1e-5, "MLE error {:.2e} exceeds 1e-5", diff_mle);

    let diff_fisher = fisher_calc
        .sub(fisher_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_fisher < 1e-5,
        "Fisher error {:.2e} exceeds 1e-5",
        diff_fisher
    );

    let diff_crlb = crlb_calc
        .sub(crlb_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_crlb < 1e-5,
        "CRLB error {:.2e} exceeds 1e-5",
        diff_crlb
    );

    let diff_lrt = lrt_calc
        .sub(lrt_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_lrt < 1e-5, "LRT error {:.2e} exceeds 1e-5", diff_lrt);

    // 3. Bayesian Conjugate & Laplace Occam Factor
    let mu_0 = tensors.get("mu_0").expect("mu_0");
    let mu_post_gold = tensors.get("mu_post").expect("mu_post");
    let sigma_post_gold = tensors.get("sigma_post_vec").expect("sigma_post_vec");
    let laplace_gold = tensors
        .get("laplace_log_marginal")
        .expect("laplace_log_marginal");
    let occam_gold = tensors.get("occam_penalty").expect("occam_penalty");

    let (mu_post_calc, sigma_post_calc, laplace_calc, occam_calc) =
        bayesian_conjugate_laplace(x_data, mu_0, 1.0, 0.5)?;

    let diff_mu_post = mu_post_calc
        .sub(mu_post_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_mu_post < 1e-5,
        "Mu post error {:.2e} exceeds 1e-5",
        diff_mu_post
    );

    let diff_sig_post = sigma_post_calc
        .sub(sigma_post_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_sig_post < 1e-5,
        "Sigma post error {:.2e} exceeds 1e-5",
        diff_sig_post
    );

    let diff_laplace = laplace_calc
        .sub(laplace_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_laplace < 1e-5,
        "Laplace error {:.2e} exceeds 1e-5",
        diff_laplace
    );

    let diff_occam = occam_calc
        .sub(occam_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_occam < 1e-5,
        "Occam error {:.2e} exceeds 1e-5",
        diff_occam
    );

    // 4. James-Stein Shrinkage
    let y_obs = tensors.get("y_obs").expect("y_obs");
    let th_js_gold = tensors.get("theta_js").expect("theta_js");
    let th_js_pos_gold = tensors.get("theta_js_pos").expect("theta_js_pos");
    let c_shrink_gold = tensors.get("c_shrink").expect("c_shrink");

    let (th_js_calc, th_js_pos_calc, c_shrink_calc) = james_stein_estimator(y_obs, 1.0)?;

    let diff_js = th_js_calc
        .sub(th_js_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_js < 1e-5,
        "James-Stein error {:.2e} exceeds 1e-5",
        diff_js
    );

    let diff_js_pos = th_js_pos_calc
        .sub(th_js_pos_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_js_pos < 1e-5,
        "James-Stein pos error {:.2e} exceeds 1e-5",
        diff_js_pos
    );

    let diff_c = c_shrink_calc
        .sub(c_shrink_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_c < 1e-5, "C shrink error {:.2e} exceeds 1e-5", diff_c);

    // 5. PAC-Bayes Bound
    let mu_q = tensors.get("mu_q").expect("mu_q");
    let sigma_q = tensors.get("sigma_q").expect("sigma_q");
    let emp_risk = tensors.get("emp_risk").expect("emp_risk");
    let kl_gold = tensors.get("kl_div").expect("kl_div");
    let pac_gold = tensors.get("pac_bound").expect("pac_bound");

    let (kl_calc, pac_calc) = pac_bayes_bound(mu_q, sigma_q, emp_risk, 16, 1.0, 0.05)?;

    let diff_kl = kl_calc
        .sub(kl_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_kl < 1e-5, "KL error {:.2e} exceeds 1e-5", diff_kl);

    let diff_pac = pac_calc
        .sub(pac_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_pac < 1e-5,
        "PAC bound error {:.2e} exceeds 1e-5",
        diff_pac
    );

    // 6. Anytime-Valid Martingale
    let e_vals = tensors.get("e_values").expect("e_values");
    let mart_gold = tensors.get("martingale").expect("martingale");
    let thresh_gold = tensors.get("ville_threshold").expect("ville_threshold");

    let (mart_calc, thresh_calc) = anytime_valid_martingale(e_vals, 0.05)?;

    let diff_mart = mart_calc
        .sub(mart_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_mart < 1e-5,
        "Martingale error {:.2e} exceeds 1e-5",
        diff_mart
    );

    let diff_thresh = thresh_calc
        .sub(thresh_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_thresh < 1e-5,
        "Threshold error {:.2e} exceeds 1e-5",
        diff_thresh
    );

    Ok(())
}
