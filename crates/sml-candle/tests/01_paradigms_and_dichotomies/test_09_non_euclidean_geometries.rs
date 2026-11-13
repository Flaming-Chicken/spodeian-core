//! Integration test for Category 01 Topic 09: Non-Euclidean Geometries (Candle Track).

use sml_candle::c01_paradigms_and_dichotomies::t09_non_euclidean_geometries_hyperbolic_poincare_spherical_and_spd::{
    poincare_distance, poincare_exp_zero, poincare_log_zero, poincare_mobius_add,
    spd_log_euclidean_distance_2x2, spherical_alignment, spherical_distance,
};
use candle_core::{safetensors::load, Device, Result};

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_09_candle_to_golden_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "01_paradigms_and_dichotomies/09_non_euclidean_geometries_hyperbolic_poincare_spherical_and_spd.safetensors",
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

    // 2. Hyperbolic Poincaré Ball Parity
    let c_val = tensors.get("layers.c").expect("c").to_vec1::<f32>()?[0] as f64;
    let v = tensors.get("layers.euclidean_v").expect("euclidean_v");
    let exp_gold = tensors.get("layers.exp_map_0").expect("exp_map_0");

    let exp_calc = poincare_exp_zero(v, c_val)?;
    let diff_exp = exp_calc
        .sub(exp_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_exp < 1e-5,
        "Poincaré exp map parity error {:.2e} exceeds 1e-5",
        diff_exp
    );

    let x_hyp = tensors.get("layers.poincare_x").expect("poincare_x");
    let log_gold = tensors.get("layers.log_map_0").expect("log_map_0");
    let log_calc = poincare_log_zero(x_hyp, c_val)?;
    let diff_log = log_calc
        .sub(log_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_log < 1e-5,
        "Poincaré log map parity error {:.2e} exceeds 1e-5",
        diff_log
    );

    let y_hyp = tensors.get("layers.poincare_y").expect("poincare_y");
    let mobius_gold = tensors.get("layers.mobius_add").expect("mobius_add");
    let mobius_calc = poincare_mobius_add(x_hyp, y_hyp, c_val)?;
    let diff_mobius = mobius_calc
        .sub(mobius_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_mobius < 1e-5,
        "Poincaré Möbius add parity error {:.2e} exceeds 1e-5",
        diff_mobius
    );

    let dist_gold = tensors.get("layers.poincare_dist").expect("poincare_dist");
    let dist_calc = poincare_distance(x_hyp, y_hyp, c_val)?;
    let diff_dist = dist_calc
        .sub(dist_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_dist < 1e-5,
        "Poincaré distance parity error {:.2e} exceeds 1e-5",
        diff_dist
    );

    // Invariant: Self-distance in Poincaré ball is 0
    let self_dist = poincare_distance(x_hyp, x_hyp, c_val)?;
    let max_self_dist = self_dist.flatten_all()?.max(0)?.to_scalar::<f32>()?;
    assert!(
        max_self_dist < 1e-5,
        "Self distance in Poincaré ball non-zero: {}",
        max_self_dist
    );

    // 3. Spherical Geometry Parity
    let p = tensors.get("layers.sphere_p").expect("sphere_p");
    let q = tensors.get("layers.sphere_q").expect("sphere_q");
    let s_dist_gold = tensors.get("layers.sphere_dist").expect("sphere_dist");
    let s_dist_calc = spherical_distance(p, q)?;
    let diff_s_dist = s_dist_calc
        .sub(s_dist_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_s_dist < 1e-5,
        "Spherical distance parity error {:.2e} exceeds 1e-5",
        diff_s_dist
    );

    let align_gold = tensors
        .get("layers.alignment_loss")
        .expect("alignment_loss");
    let align_calc = spherical_alignment(p, q)?;
    let diff_align = align_calc
        .sub(align_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_align < 1e-5,
        "Spherical alignment loss parity error {:.2e} exceeds 1e-5",
        diff_align
    );

    // 4. SPD Log-Euclidean Distance Parity
    let spd_p = tensors.get("layers.spd_P").expect("spd_P");
    let spd_q = tensors.get("layers.spd_Q").expect("spd_Q");
    let spd_dist_gold = tensors
        .get("layers.log_euclidean_dist")
        .expect("log_euclidean_dist");

    let spd_dist_calc = spd_log_euclidean_distance_2x2(spd_p, spd_q)?;
    let diff_spd = spd_dist_calc
        .sub(spd_dist_gold)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        diff_spd < 1e-5,
        "SPD Log-Euclidean distance parity error {:.2e} exceeds 1e-5",
        diff_spd
    );

    Ok(())
}
