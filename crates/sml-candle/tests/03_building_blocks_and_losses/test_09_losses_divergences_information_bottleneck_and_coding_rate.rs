//! Integration test for Category 03 Topic 09: Losses Divergences, Information Bottleneck, and Coding Rate (Candle Track).

use ai_candle::c03_building_blocks_and_losses::t09_losses_divergences_information_bottleneck_and_coding_rate::{
    itakura_saito_forward, jsd_forward, kl_forward, kl_reverse, mcr2_coding_rate_forward,
    renyi_divergence_forward, vib_kl_forward, LinearAnchor,
};
use candle_core::{safetensors::load, Device, Result};

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_09_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "03_building_blocks_and_losses/09_losses_divergences_information_bottleneck_and_coding_rate.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let test_in = tensors.get("test_input").expect("test_input");
    let test_out = tensors.get("test_output").expect("test_output");
    let weight = tensors.get("weight").expect("weight");
    let bias = tensors.get("bias").expect("bias");

    let p_dist = tensors.get("p_dist").expect("p_dist");
    let q_dist = tensors.get("q_dist").expect("q_dist");
    let loss_fwd_kl_gold = tensors
        .get("loss_fwd_kl")
        .expect("loss_fwd_kl")
        .to_vec1::<f32>()?[0];
    let loss_rev_kl_gold = tensors
        .get("loss_rev_kl")
        .expect("loss_rev_kl")
        .to_vec1::<f32>()?[0];
    let loss_jsd_gold = tensors
        .get("loss_jsd")
        .expect("loss_jsd")
        .to_vec1::<f32>()?[0];

    let r_alpha = tensors
        .get("renyi_alpha")
        .expect("renyi_alpha")
        .to_vec1::<f32>()?[0] as f64;
    let loss_renyi_gold = tensors
        .get("loss_renyi")
        .expect("loss_renyi")
        .to_vec1::<f32>()?[0];

    let is_x = tensors.get("is_x").expect("is_x");
    let is_y = tensors.get("is_y").expect("is_y");
    let loss_is_gold = tensors.get("loss_is").expect("loss_is").to_vec1::<f32>()?[0];

    let vib_mu = tensors.get("vib_mu").expect("vib_mu");
    let vib_logvar = tensors.get("vib_logvar").expect("vib_logvar");
    let loss_vib_gold = tensors
        .get("loss_vib")
        .expect("loss_vib")
        .to_vec1::<f32>()?[0];

    let mcr2_z = tensors.get("mcr2_z").expect("mcr2_z");
    let mcr2_eps = tensors
        .get("mcr2_eps")
        .expect("mcr2_eps")
        .to_vec1::<f32>()?[0] as f64;
    let loss_mcr2_gold = tensors
        .get("loss_mcr2")
        .expect("loss_mcr2")
        .to_vec1::<f32>()?[0];

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

    // 2. Forward KL Parity
    let loss_fwd_calc = kl_forward(p_dist, q_dist, 1e-12)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let diff_fwd = (loss_fwd_calc - loss_fwd_kl_gold).abs();
    assert!(
        diff_fwd < 1e-5,
        "Forward KL parity error {:.2e} exceeds 1e-5",
        diff_fwd
    );

    // 3. Reverse KL Parity
    let loss_rev_calc = kl_reverse(p_dist, q_dist, 1e-12)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let diff_rev = (loss_rev_calc - loss_rev_kl_gold).abs();
    assert!(
        diff_rev < 1e-5,
        "Reverse KL parity error {:.2e} exceeds 1e-5",
        diff_rev
    );

    // 4. JSD Parity
    let loss_jsd_calc = jsd_forward(p_dist, q_dist)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let diff_jsd = (loss_jsd_calc - loss_jsd_gold).abs();
    assert!(
        diff_jsd < 1e-5,
        "JSD parity error {:.2e} exceeds 1e-5",
        diff_jsd
    );

    // 5. Renyi Parity
    let loss_renyi_calc = renyi_divergence_forward(p_dist, q_dist, r_alpha, 1e-12)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let diff_renyi = (loss_renyi_calc - loss_renyi_gold).abs();
    assert!(
        diff_renyi < 1e-5,
        "Renyi parity error {:.2e} exceeds 1e-5",
        diff_renyi
    );

    // 6. Itakura-Saito Parity
    let loss_is_calc = itakura_saito_forward(is_x, is_y, 1e-12)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let diff_is = (loss_is_calc - loss_is_gold).abs();
    assert!(
        diff_is < 1e-5,
        "Itakura-Saito parity error {:.2e} exceeds 1e-5",
        diff_is
    );

    // 7. VIB KL Parity
    let loss_vib_calc = vib_kl_forward(vib_mu, vib_logvar)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let diff_vib = (loss_vib_calc - loss_vib_gold).abs();
    assert!(
        diff_vib < 1e-5,
        "VIB KL parity error {:.2e} exceeds 1e-5",
        diff_vib
    );

    // 8. MCR^2 Coding Rate Parity
    let loss_mcr2_calc = mcr2_coding_rate_forward(mcr2_z, mcr2_eps)?
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let diff_mcr2 = (loss_mcr2_calc - loss_mcr2_gold).abs();
    assert!(
        diff_mcr2 < 1e-5,
        "MCR^2 parity error {:.2e} exceeds 1e-5",
        diff_mcr2
    );

    Ok(())
}
