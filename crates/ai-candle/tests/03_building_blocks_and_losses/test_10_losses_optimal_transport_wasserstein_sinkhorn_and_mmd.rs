//! Integration test for Category 03 Topic 10: Losses Optimal Transport, Wasserstein, Sinkhorn, and MMD (Candle Track).

use ai_candle::c03_building_blocks_and_losses::t10_losses_optimal_transport_wasserstein_sinkhorn_and_mmd::{
    debiased_sinkhorn_forward, energy_distance_forward, mmd_forward,
    sinkhorn_cost_forward, sliced_wasserstein_forward, wasserstein_1d_forward, LinearAnchor,
};
use candle_core::{safetensors::load, Device, Result};

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_10_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "03_building_blocks_and_losses/10_losses_optimal_transport_wasserstein_sinkhorn_and_mmd.safetensors",
    );
    let tensors = load(weights_path, &device)?;

    let test_in = tensors.get("test_input").expect("test_input");
    let test_out = tensors.get("test_output").expect("test_output");
    let weight = tensors.get("weight").expect("weight");
    let bias = tensors.get("bias").expect("bias");

    let u_1d = tensors.get("u_1d").expect("u_1d");
    let v_1d = tensors.get("v_1d").expect("v_1d");
    let expected_w1 = tensors
        .get("expected_w1")
        .expect("expected_w1")
        .to_vec1::<f32>()?[0];

    let swd_x = tensors.get("swd_x").expect("swd_x");
    let swd_y = tensors.get("swd_y").expect("swd_y");
    let projections = tensors.get("projections").expect("projections");
    let expected_swd = tensors
        .get("expected_swd")
        .expect("expected_swd")
        .to_vec1::<f32>()?[0];

    let cloud_p = tensors.get("cloud_p").expect("cloud_p");
    let cloud_q = tensors.get("cloud_q").expect("cloud_q");
    let expected_sink_raw = tensors
        .get("expected_sinkhorn_raw")
        .expect("expected_sinkhorn_raw")
        .to_vec1::<f32>()?[0];
    let expected_sink_deb = tensors
        .get("expected_sinkhorn_debiased")
        .expect("expected_sinkhorn_debiased")
        .to_vec1::<f32>()?[0];
    let expected_mmd = tensors
        .get("expected_mmd")
        .expect("expected_mmd")
        .to_vec1::<f32>()?[0];
    let expected_energy = tensors
        .get("expected_energy")
        .expect("expected_energy")
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

    // 2. 1D Wasserstein Parity
    let w1_pred = wasserstein_1d_forward(u_1d, v_1d)?;
    let w1_val = w1_pred.to_vec1::<f32>()?[0];
    let diff_w1 = (w1_val - expected_w1).abs();
    assert!(
        diff_w1 < 1e-5,
        "1D Wasserstein parity error {:.2e} exceeds 1e-5",
        diff_w1
    );

    // 3. Sliced Wasserstein Parity
    let swd_pred = sliced_wasserstein_forward(swd_x, swd_y, projections)?;
    let swd_val = swd_pred.to_vec1::<f32>()?[0];
    let diff_swd = (swd_val - expected_swd).abs();
    assert!(
        diff_swd < 1e-5,
        "SWD parity error {:.2e} exceeds 1e-5",
        diff_swd
    );

    // 4. Sinkhorn Raw Parity
    let sink_raw_pred = sinkhorn_cost_forward(cloud_p, cloud_q, 0.5f32, 20)?;
    let sink_raw_val = sink_raw_pred.to_vec1::<f32>()?[0];
    let diff_sink_raw = (sink_raw_val - expected_sink_raw).abs();
    assert!(
        diff_sink_raw < 1e-5,
        "Sinkhorn raw parity error {:.2e} exceeds 1e-5",
        diff_sink_raw
    );

    // 5. Debiased Sinkhorn Parity
    let sink_deb_pred = debiased_sinkhorn_forward(cloud_p, cloud_q, 0.5f32, 20)?;
    let sink_deb_val = sink_deb_pred.to_vec1::<f32>()?[0];
    let diff_sink_deb = (sink_deb_val - expected_sink_deb).abs();
    assert!(
        diff_sink_deb < 1e-5,
        "Debiased Sinkhorn parity error {:.2e} exceeds 1e-5",
        diff_sink_deb
    );

    // 6. MMD Parity
    let mmd_pred = mmd_forward(cloud_p, cloud_q, &[0.5f32, 1.0f32, 2.0f32])?;
    let mmd_val = mmd_pred.to_vec1::<f32>()?[0];
    let diff_mmd = (mmd_val - expected_mmd).abs();
    assert!(
        diff_mmd < 1e-5,
        "MMD parity error {:.2e} exceeds 1e-5",
        diff_mmd
    );

    // 7. Energy Distance Parity
    let energy_pred = energy_distance_forward(cloud_p, cloud_q)?;
    let energy_val = energy_pred.to_vec1::<f32>()?[0];
    let diff_energy = (energy_val - expected_energy).abs();
    assert!(
        diff_energy < 1e-5,
        "Energy distance parity error {:.2e} exceeds 1e-5",
        diff_energy
    );

    Ok(())
}
