use candle_core::{Device, Result, safetensors::load};

use sml_candle::c05_architectures::t02_mixture_of_experts_top_k_expert_choice_and_soft_moe as moe;

#[path = "../common/mod.rs"]
mod common;

#[test]
fn test_02_mixture_of_experts_top_k_expert_choice_and_soft_moe_parity() -> Result<()> {
    let device = Device::Cpu;
    let weights_path = common::get_weights_path(
        "05_architectures/01_mlp_moe_soups_and_kan/02_mixture_of_experts_top_k_expert_choice_and_soft_moe.safetensors",
    );
    let weights = load(weights_path, &device)?;

    let x = weights.get("input").unwrap();
    let router_w = weights.get("router_weight").unwrap();
    let router_b = weights.get("router_bias").unwrap();
    let experts_w1 = weights.get("experts_w1").unwrap();
    let experts_b1 = weights.get("experts_b1").unwrap();
    let experts_w2 = weights.get("experts_w2").unwrap();
    let experts_b2 = weights.get("experts_b2").unwrap();

    // 1. Top-K MoE
    let (topk_out, aux_loss, z_loss) = moe::topk_moe_forward(
        x, router_w, router_b, experts_w1, experts_b1, experts_w2, experts_b2, 2,
    )?;
    let expected_topk = weights.get("topk_output").unwrap();
    let diff_topk = topk_out
        .sub(expected_topk)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_topk < 1e-5, "Top-K MoE error: {}", diff_topk);

    let expected_aux = weights
        .get("aux_loss")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let diff_aux = (aux_loss - expected_aux).abs();
    assert!(diff_aux < 1e-5, "Aux loss error: {}", diff_aux);

    let expected_z = weights
        .get("z_loss")
        .unwrap()
        .flatten_all()?
        .to_vec1::<f32>()?[0];
    let diff_z = (z_loss - expected_z).abs();
    assert!(diff_z < 1e-5, "Z loss error: {}", diff_z);

    // 2. Expert Choice MoE
    let ec_out = moe::expert_choice_moe_forward(
        x, router_w, router_b, experts_w1, experts_b1, experts_w2, experts_b2, 2,
    )?;
    let expected_ec = weights.get("expert_choice_output").unwrap();
    let diff_ec = ec_out
        .sub(expected_ec)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_ec < 1e-5, "Expert Choice MoE error: {}", diff_ec);

    // 3. Soft MoE
    let soft_disp_w = weights.get("soft_disp_weight").unwrap();
    let soft_disp_b = weights.get("soft_disp_bias").unwrap();
    let soft_comb_w = weights.get("soft_comb_weight").unwrap();
    let soft_comb_b = weights.get("soft_comb_bias").unwrap();

    let soft_out = moe::soft_moe_forward(
        x,
        soft_disp_w,
        soft_disp_b,
        soft_comb_w,
        soft_comb_b,
        experts_w1,
        experts_b1,
        experts_w2,
        experts_b2,
        1,
    )?;
    let expected_soft = weights.get("soft_moe_output").unwrap();
    let diff_soft = soft_out
        .sub(expected_soft)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff_soft < 1e-5, "Soft MoE error: {}", diff_soft);

    // 4. Anchor Parity
    let test_in = weights.get("test_input").unwrap();
    let weight = weights.get("weight").unwrap();
    let bias = weights.get("bias").unwrap();
    let anchor_out = moe::linear_anchor_forward(test_in, weight, bias)?;
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
