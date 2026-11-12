use ai_candle::c02_classical_and_probabilistic_ml::t12_mcmc_hmc_sgld_and_stein_variational_gradient_descent_svgd as mcmc;
use candle_core::{Device, Result};
use std::path::PathBuf;

#[test]
fn test_12_mcmc_hmc_sgld_and_stein_variational_gradient_descent_svgd() -> Result<()> {
    let device = Device::Cpu;

    // 1. Load Reference Safetensors Checkpoint
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let weights_path = manifest_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("weights/02_classical_and_probabilistic_ml/12_mcmc_hmc_sgld_and_stein_variational_gradient_descent_svgd.safetensors");

    if !weights_path.exists() {
        eprintln!(
            "Skipping test: weights file not found at {:?}",
            weights_path
        );
        return Ok(());
    }

    let tensors = candle_core::safetensors::load(&weights_path, &device)?;

    // 2. Parity Check: HMC Leapfrog Integration
    let theta_init = tensors.get("theta_init").expect("Missing theta_init");
    let p_init = tensors.get("p_init").expect("Missing p_init");
    let sigma_inv = tensors.get("Sigma_inv").expect("Missing Sigma_inv");
    let golden_hmc_theta = tensors
        .get("hmc_theta_prop")
        .expect("Missing hmc_theta_prop");
    let golden_hmc_p = tensors.get("hmc_p_prop").expect("Missing hmc_p_prop");
    let golden_hmc_accept = tensors
        .get("hmc_accept_ratio")
        .expect("Missing hmc_accept_ratio");

    let (hmc_theta, hmc_p, hmc_accept) = mcmc::hmc_leapfrog(theta_init, p_init, sigma_inv, 0.1, 5)?;
    let diff_theta = (&hmc_theta - golden_hmc_theta)?
        .abs()?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_p = (&hmc_p - golden_hmc_p)?.abs()?.max(0)?.to_scalar::<f32>()?;
    let diff_accept = (&hmc_accept - golden_hmc_accept)?
        .abs()?
        .max(0)?
        .to_scalar::<f32>()?;

    assert!(diff_theta < 1e-4, "HMC theta diff {diff_theta:.2e} >= 1e-4");
    assert!(diff_p < 1e-4, "HMC momentum diff {diff_p:.2e} >= 1e-4");
    assert!(
        diff_accept < 1e-4,
        "HMC accept ratio diff {diff_accept:.2e} >= 1e-4"
    );

    // 3. Parity Check: SGLD Step
    let theta_sgld = tensors.get("theta_sgld").expect("Missing theta_sgld");
    let grad_logp_sgld = tensors
        .get("grad_logp_sgld")
        .expect("Missing grad_logp_sgld");
    let noise_sgld = tensors.get("noise_sgld").expect("Missing noise_sgld");
    let golden_sgld_next = tensors
        .get("theta_sgld_next")
        .expect("Missing theta_sgld_next");

    let sgld_next = mcmc::sgld_step(theta_sgld, grad_logp_sgld, noise_sgld, 0.05)?;
    let diff_sgld = (&sgld_next - golden_sgld_next)?
        .abs()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_sgld < 1e-4, "SGLD step diff {diff_sgld:.2e} >= 1e-4");

    // 4. Parity Check: SVGD Velocity & Kernel Gram Matrix
    let particles_init = tensors
        .get("particles_init")
        .expect("Missing particles_init");
    let grad_logp_svgd = tensors
        .get("grad_logp_svgd")
        .expect("Missing grad_logp_svgd");
    let golden_svgd_vel = tensors.get("svgd_velocity").expect("Missing svgd_velocity");
    let golden_kernel_gram = tensors
        .get("svgd_kernel_gram")
        .expect("Missing svgd_kernel_gram");
    let golden_particles_next = tensors
        .get("particles_next")
        .expect("Missing particles_next");

    let (svgd_vel, k_gram) = mcmc::svgd_velocity_field(particles_init, grad_logp_svgd, 1.0)?;
    let diff_vel = (&svgd_vel - golden_svgd_vel)?
        .abs()?
        .max(0)?
        .max(0)?
        .to_scalar::<f32>()?;
    let diff_k = (&k_gram - golden_kernel_gram)?
        .abs()?
        .max(0)?
        .max(0)?
        .to_scalar::<f32>()?;

    assert!(diff_vel < 1e-4, "SVGD velocity diff {diff_vel:.2e} >= 1e-4");
    assert!(diff_k < 1e-4, "SVGD kernel gram diff {diff_k:.2e} >= 1e-4");

    let particles_next = (particles_init + svgd_vel.affine(0.1, 0.0)?)?;
    let diff_part = (&particles_next - golden_particles_next)?
        .abs()?
        .max(0)?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_part < 1e-4,
        "SVGD particles next diff {diff_part:.2e} >= 1e-4"
    );

    Ok(())
}
