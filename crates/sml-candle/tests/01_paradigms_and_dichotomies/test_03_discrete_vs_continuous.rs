//! Integration test for Category 01 Topic 03: Discrete vs Continuous Latents & Gradient Estimators (Candle Track).

use ai_candle::c01_paradigms_and_dichotomies::t03_discrete_vs_continuous_latents_and_gradient_estimators::{
    GaussianReparameterization, GumbelSoftmaxRelaxation,
};
use candle_core::{safetensors::load, Device, Result, Tensor};

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_03_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "01_paradigms_and_dichotomies/03_discrete_vs_continuous_latents_and_gradient_estimators.safetensors",
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

    // 2. Continuous Pathwise Reparameterization Parity
    let w_mu = tensors.get("layers.w_mu").expect("w_mu");
    let b_mu = tensors.get("layers.b_mu").expect("b_mu");
    let w_lv = tensors.get("layers.w_logvar").expect("w_logvar");
    let b_lv = tensors.get("layers.b_logvar").expect("b_logvar");
    let w_dec = tensors.get("layers.w_dec").expect("w_dec");
    let b_dec = tensors.get("layers.b_dec").expect("b_dec");
    let eps = tensors.get("layers.eps").expect("eps");

    let continuous_z_gold = tensors.get("layers.continuous_z").expect("continuous_z");
    let continuous_out_gold = tensors
        .get("layers.continuous_out")
        .expect("continuous_out");
    let kl_gold = tensors.get("layers.kl_divergence").expect("kl_divergence");

    let vae = GaussianReparameterization::new(
        w_mu.clone(),
        b_mu.clone(),
        w_lv.clone(),
        b_lv.clone(),
        w_dec.clone(),
        b_dec.clone(),
    );
    let (z_calc, out_calc, kl_calc) = vae.forward(test_in, eps)?;

    let diff_z = z_calc
        .sub(continuous_z_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_z < 1e-5,
        "Continuous latent z parity error {:.2e} exceeds 1e-5",
        diff_z
    );

    let diff_out = out_calc
        .sub(continuous_out_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_out < 1e-5,
        "Continuous output parity error {:.2e} exceeds 1e-5",
        diff_out
    );

    let diff_kl = kl_calc
        .sub(kl_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_kl < 1e-5,
        "Gaussian KL divergence error {:.2e} exceeds 1e-5",
        diff_kl
    );

    // 3. Discrete Gumbel-Softmax Parity
    let w_logits = tensors.get("layers.w_logits").expect("w_logits");
    let b_logits = tensors.get("layers.b_logits").expect("b_logits");
    let gumbel_noise = tensors.get("layers.gumbel_noise").expect("gumbel_noise");
    let gumbel_soft_gold = tensors.get("layers.gumbel_soft").expect("gumbel_soft");
    let gumbel_out_gold = tensors.get("layers.gumbel_out").expect("gumbel_out");
    let temp = tensors
        .get("layers.temperature")
        .expect("temperature")
        .to_vec1::<f32>()?[0] as f64;

    let gs_model = GumbelSoftmaxRelaxation::new(
        w_logits.clone(),
        b_logits.clone(),
        w_dec.clone(),
        b_dec.clone(),
        temp,
    );
    let (gs_soft_calc, gs_out_calc) = gs_model.forward(test_in, gumbel_noise)?;

    let diff_gs_soft = gs_soft_calc
        .sub(gumbel_soft_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_gs_soft < 1e-5,
        "Gumbel soft sample parity error {:.2e} exceeds 1e-5",
        diff_gs_soft
    );

    let diff_gs_out = gs_out_calc
        .sub(gumbel_out_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_gs_out < 1e-5,
        "Gumbel output parity error {:.2e} exceeds 1e-5",
        diff_gs_out
    );

    // 4. Mathematical Invariant: KL divergence of standard normal is zero
    let zero_mu = Tensor::zeros_like(w_mu)?;
    let zero_lv = Tensor::zeros_like(w_lv)?;
    let kl_zero = GaussianReparameterization::kl_divergence(&zero_mu, &zero_lv)?;
    let kl_zero_val = kl_zero.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
    assert!(
        kl_zero_val < 1e-6,
        "KL divergence at N(0, I) must be exactly zero, got {:.2e}",
        kl_zero_val
    );

    Ok(())
}
