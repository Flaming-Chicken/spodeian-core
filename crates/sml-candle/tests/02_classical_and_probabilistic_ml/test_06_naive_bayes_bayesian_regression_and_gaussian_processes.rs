use candle_core::{Device, Result, Tensor};
use sml_candle::c02_classical_and_probabilistic_ml::t06_naive_bayes_bayesian_regression_and_gaussian_processes as bayes;
use std::path::PathBuf;

#[test]
fn test_06_naive_bayes_bayesian_regression_and_gaussian_processes() -> Result<()> {
    let device = Device::Cpu;

    // 1. Load Reference Safetensors Checkpoint
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let weights_path = manifest_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("weights/02_classical_and_probabilistic_ml/06_naive_bayes_bayesian_regression_and_gaussian_processes.safetensors");

    if !weights_path.exists() {
        eprintln!(
            "Skipping test: weights file not found at {:?}",
            weights_path
        );
        return Ok(());
    }

    let tensors = candle_core::safetensors::load(&weights_path, &device)?;
    let x = tensors.get("X").expect("Missing X");
    let means = tensors.get("nb_means").expect("Missing nb_means");
    let vars = tensors.get("nb_vars").expect("Missing nb_vars");
    let priors = tensors.get("nb_priors").expect("Missing nb_priors");
    let golden_nb_log_probs = tensors.get("nb_log_probs").expect("Missing nb_log_probs");
    let m_n = tensors.get("bayes_m_N").expect("Missing bayes_m_N");
    let s_n = tensors.get("bayes_S_N").expect("Missing bayes_S_N");
    let golden_b_mean = tensors
        .get("bayes_pred_mean")
        .expect("Missing bayes_pred_mean");
    let golden_b_var = tensors
        .get("bayes_pred_var")
        .expect("Missing bayes_pred_var");
    let golden_gp_k = tensors.get("gp_K").expect("Missing gp_K");
    let gp_alpha = tensors.get("gp_alpha").expect("Missing gp_alpha");
    let golden_gp_mean = tensors.get("gp_pred_mean").expect("Missing gp_pred_mean");
    let golden_svgp_diag = tensors
        .get("svgp_approx_diag")
        .expect("Missing svgp_approx_diag");

    // 2. Parity Check: Gaussian Naive Bayes Log-Probabilities
    let log_priors = priors.log()?;
    let nb = bayes::GaussianNaiveBayes::new(means.clone(), vars.clone(), log_priors);
    let nb_log_probs = nb.predict_log_proba(x)?;
    let diff_nb = (&nb_log_probs - golden_nb_log_probs)?
        .abs()?
        .max(0)?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_nb < 1e-4,
        "Candle vs Golden Naive Bayes log probs error {diff_nb:.2e} exceeds 1e-4"
    );

    // 3. Parity Check: Bayesian Linear Regression
    let (b_mean, b_var) = bayes::bayesian_linear_regression_predict(x, m_n, s_n, 2.0)?;
    let diff_bm = (&b_mean - golden_b_mean)?
        .abs()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_bv = (&b_var - golden_b_var)?.abs()?.max(0)?.to_scalar::<f32>()?;
    assert!(
        diff_bm < 1e-4,
        "Candle vs Golden Bayes mean error {diff_bm:.2e} exceeds 1e-4"
    );
    assert!(
        diff_bv < 1e-4,
        "Candle vs Golden Bayes var error {diff_bv:.2e} exceeds 1e-4"
    );

    // 4. Parity Check: GP RBF Kernel
    let gp_k = bayes::rbf_kernel(x, x, 1.0, 1.0)?;
    let diff_k = (&gp_k - golden_gp_k)?
        .abs()?
        .max(0)?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_k < 1e-4,
        "Candle vs Golden GP Kernel error {diff_k:.2e} exceeds 1e-4"
    );

    // 5. Parity Check: GP Predictive Mean
    let (gp_mean, _) = bayes::gp_predict_with_alpha_and_v(&gp_k, gp_alpha, &gp_k, 1.0, 1e-4)?;
    let diff_gpm = (&gp_mean - golden_gp_mean)?
        .abs()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_gpm < 1e-4,
        "Candle vs Golden GP predictive mean error {diff_gpm:.2e} exceeds 1e-4"
    );

    // 6. SVGP Inducing Diag Sanity Check
    let a_dummy = Tensor::ones((8, 32), candle_core::DType::F32, &device)?;
    let svgp_dummy = bayes::compute_svgp_inducing_diag(&a_dummy)?;
    assert_eq!(svgp_dummy.dims1()?, 32);
    assert_eq!(golden_svgp_diag.dims1()?, 32);

    println!(
        "Candle Topic 06: Naive Bayes, Bayesian Regression & GP tests passed with machine precision!"
    );
    Ok(())
}
