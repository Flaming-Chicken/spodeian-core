use candle_core::{Device, Result, safetensors::load};

use ai_candle::c05_architectures::t04_soup_of_experts_smear_rewarded_soups_btx_and_upcycling as smear;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_04_soup_of_experts_smear_rewarded_soups_btx_and_upcycling_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "05_architectures/01_mlp_moe_soups_and_kan/04_soup_of_experts_smear_rewarded_soups_btx_and_upcycling.safetensors",
    );
    let weights = load(weights_path, &device)?;

    let x = weights.get("input").unwrap();
    let experts_w1 = weights.get("experts_w1").unwrap();
    let experts_b1 = weights.get("experts_b1").unwrap();
    let experts_w2 = weights.get("experts_w2").unwrap();
    let experts_b2 = weights.get("experts_b2").unwrap();
    let router_w = weights.get("router_weight").unwrap();
    let router_b = weights.get("router_bias").unwrap();

    // 1. Soup of Experts (SMEAR) Forward
    let (smear_out, _) = smear::soup_of_experts_smear_forward(
        x, experts_w1, experts_b1, experts_w2, experts_b2, router_w, router_b,
    )?;
    let expected_smear = weights.get("smear_output").unwrap();
    let diff_smear = smear_out
        .sub(expected_smear)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_smear < 1e-5, "SMEAR error: {}", diff_smear);

    // 2. Activation-Space MoE Forward
    let act_out = smear::activation_moe_forward(
        x, experts_w1, experts_b1, experts_w2, experts_b2, router_w, router_b,
    )?;
    let expected_act = weights.get("act_output").unwrap();
    let diff_act = act_out
        .sub(expected_act)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_act < 1e-5, "Activation MoE error: {}", diff_act);

    // 3. Rewarded Pareto Soup Forward
    let pareto_out =
        smear::evaluate_pareto_soup(x, experts_w1, experts_b1, experts_w2, experts_b2, 0.6)?;
    let expected_pareto = weights.get("pareto_soup_output").unwrap();
    let diff_pareto = pareto_out
        .sub(expected_pareto)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_pareto < 1e-5, "Pareto Soup error: {}", diff_pareto);

    // 4. Anchor Parity
    let test_in = weights.get("test_input").unwrap();
    let weight = weights.get("weight").unwrap();
    let bias = weights.get("bias").unwrap();
    let anchor_out = smear::linear_anchor_forward(test_in, weight, bias)?;
    let expected_anchor = weights.get("test_output").unwrap();
    let diff_anchor = anchor_out
        .sub(expected_anchor)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_anchor < 1e-5, "Linear anchor error: {}", diff_anchor);

    Ok(())
}
