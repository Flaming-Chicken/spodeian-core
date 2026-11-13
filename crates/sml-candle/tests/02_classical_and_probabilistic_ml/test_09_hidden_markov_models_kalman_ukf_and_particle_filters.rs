use candle_core::{Device, Result, Tensor};
use sml_candle::c02_classical_and_probabilistic_ml::t09_hidden_markov_models_kalman_ukf_and_particle_filters as kf;
use std::path::PathBuf;

#[test]
fn test_09_hidden_markov_models_kalman_ukf_and_particle_filters() -> Result<()> {
    let device = Device::Cpu;

    // 1. Load Reference Safetensors Checkpoint
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let weights_path = manifest_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("weights/02_classical_and_probabilistic_ml/09_hidden_markov_models_kalman_ukf_and_particle_filters.safetensors");

    if !weights_path.exists() {
        eprintln!(
            "Skipping test: weights file not found at {:?}",
            weights_path
        );
        return Ok(());
    }

    let tensors = candle_core::safetensors::load(&weights_path, &device)?;

    // 2. Kalman Filter Parity Check
    let f = tensors.get("F").expect("Missing F");
    let h = tensors.get("H").expect("Missing H");
    let q = tensors.get("Q").expect("Missing Q");
    let r = tensors.get("R").expect("Missing R");
    let y = tensors.get("Y").expect("Missing Y");
    let golden_x_filt = tensors.get("x_filt").expect("Missing x_filt");
    let golden_p_diag = tensors.get("P_filt_diag").expect("Missing P_filt_diag");

    let x0 = Tensor::new(&[0.0f32, 0.0f32], &device)?;
    let p0 = Tensor::new(&[[1.0f32, 0.0f32], [0.0f32, 1.0f32]], &device)?;

    let (x_filt, p_diag) = kf::kalman_filter_forward(y, f, h, q, r, &x0, &p0)?;

    let diff_kf_x = (&x_filt - golden_x_filt)?
        .abs()?
        .max(0)?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_kf_p = (&p_diag - golden_p_diag)?
        .abs()?
        .max(0)?
        .max(0)?
        .to_scalar::<f32>()?;

    assert!(
        diff_kf_x < 1e-4,
        "Candle vs Golden Kalman x_filt error {diff_kf_x:.2e} exceeds 1e-4"
    );
    assert!(
        diff_kf_p < 1e-4,
        "Candle vs Golden Kalman P_diag error {diff_kf_p:.2e} exceeds 1e-4"
    );

    // 3. HMM Forward & Viterbi Parity Check
    let log_pi = tensors.get("log_pi").expect("Missing log_pi");
    let log_a = tensors.get("log_A").expect("Missing log_A");
    let log_b = tensors.get("log_B").expect("Missing log_B");
    let obs_seq = tensors.get("obs_seq").expect("Missing obs_seq");
    let golden_log_prob = tensors.get("hmm_log_prob").expect("Missing hmm_log_prob");
    let golden_viterbi = tensors.get("viterbi_path").expect("Missing viterbi_path");

    let log_prob = kf::hmm_forward_log_prob(obs_seq, log_pi, log_a, log_b)?;
    let diff_ll = (&log_prob - golden_log_prob)?
        .abs()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_ll < 1e-4,
        "Candle vs Golden HMM log prob error {diff_ll:.2e} exceeds 1e-4"
    );

    let viterbi_path = kf::hmm_viterbi_decode(obs_seq, log_pi, log_a, log_b)?;
    let path_diff = (&viterbi_path - golden_viterbi)?
        .abs()?
        .max(0)?
        .to_scalar::<i64>()?;
    assert_eq!(path_diff, 0, "Candle vs Golden HMM Viterbi path mismatch");

    // 4. UKF Sigma Points Parity Check
    let mu_ukf = tensors.get("mu_ukf").expect("Missing mu_ukf");
    let sigma_ukf = tensors.get("Sigma_ukf").expect("Missing Sigma_ukf");
    let golden_sigma_pts = tensors.get("sigma_pts").expect("Missing sigma_pts");

    let sigma_pts = kf::generate_ukf_sigma_points(mu_ukf, sigma_ukf, 1e-3, 2.0, 0.0)?;
    let diff_ukf = (&sigma_pts - golden_sigma_pts)?
        .abs()?
        .max(0)?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_ukf < 1e-4,
        "Candle vs Golden UKF sigma points error {diff_ukf:.2e} exceeds 1e-4"
    );

    println!("Candle Topic 09: Kalman Filter, HMM & UKF tests passed with machine precision!");
    Ok(())
}
